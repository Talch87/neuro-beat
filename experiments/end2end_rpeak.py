"""End-to-end ablation: run the frozen cascade on DETECTED R-peaks, not annotations.

The paper's headline VEB result assumes ground-truth R-peak locations. This script
removes that assumption on DS2: for each record it detects R-peaks, segments beats at
the DETECTED peaks, recomputes RR features from the detected-peak spacing, re-runs the
real screener + three-seed confirmer cascade on those windows, and scores VEB
detection against the true beat labels -- so QRS misses become end-to-end false
negatives and spurious detections become end-to-end false positives.

We run two detectors to show how much the R-peak stage matters:
  - xqrs: wfdb's standard XQRS detector (a realistic reference detector)
  - light: the repo's minimal on-patch detector (neurocardio.stream.qrs.find_r_peaks),
    a static-threshold Pan-Tompkins variant that is cheap but weaker

Nothing is retrained; the frozen screener/confirmer weights and biases are used. RR
features are standardized with the same train statistics as the locked pipeline.

Usage:
  python experiments/end2end_rpeak.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import lock_snn_rr as L  # noqa: E402
import gated_ensemble_veb as g  # noqa: E402
from lock_snn_rr import TRAIN_RECORDS, raw_group  # noqa: E402
from neurocardio.config import Config  # noqa: E402
from neurocardio.data.preprocess import bandpass_filter, normalize  # noqa: E402
from neurocardio.data.records import load_record  # noqa: E402
from neurocardio.data.segment import symbol_to_aami, _CLASS_INDEX  # noqa: E402
from neurocardio.data.splits import DS2_RECORDS  # noqa: E402
from neurocardio.encoding.beat import encode_beat  # noqa: E402
from neurocardio.models.snn import SNNClassifier  # noqa: E402
from neurocardio.stream.qrs import find_r_peaks  # noqa: E402
from neurocardio.train.loop import resolve_device  # noqa: E402
from wfdb.processing import xqrs_detect  # noqa: E402

VEB = 2
ORDERS = [0, 1]
CONF_THR, CONF_T, CONF_H = 0.12, 64, 128
SCR_THR, SCR_T, SCR_H = g.SCREENER["threshold"], g.SCREENER["n_timesteps"], g.SCREENER["hidden"]
TOL_S = 0.15  # EC57 R-peak match tolerance (150 ms)


def greedy_match(detected, true_samp, tol):
    """Assign each detected peak to the nearest unused true beat within tol.
    Returns det_to_true [-1 = false detection] and used[true] boolean."""
    used = np.zeros(len(true_samp), bool)
    det_to_true = np.full(len(detected), -1)
    for i, d in enumerate(detected):
        j = int(np.searchsorted(true_samp, d))
        best, bd = -1, tol + 1
        for k in (j - 1, j):
            if 0 <= k < len(true_samp) and not used[k]:
                dist = abs(true_samp[k] - d)
                if dist < bd:
                    bd, best = dist, k
        if best >= 0 and bd <= tol:
            used[best] = True
            det_to_true[i] = best
    return det_to_true, used


def main():
    device = resolve_device("auto")
    cfg = Config()
    wb, wa = cfg.data.window_before, cfg.data.window_after
    tol = int(TOL_S * cfg.data.fs)

    # train RR standardization (identical to the locked pipeline)
    _, _, trr = raw_group(cfg, TRAIN_RECORDS, "trainsub")
    mu, sd = trr.mean(0), trr.std(0) + 1e-6

    # frozen models + biases
    gj = json.loads(Path("models/neurobeat-veb-v1/gated_ensemble.json").read_text())
    b1 = np.array(gj["screener"]["bias"]); b2 = np.array(gj["confirmer"]["bias"])
    inf = 2 * len(ORDERS)
    screener = SNNClassifier(in_features=inf, hidden=SCR_H, n_classes=5, n_rr=3).to(device)
    screener.load_state_dict(torch.load("models/neurobeat-veb-v1/screener_weights.pt",
                                        map_location=device))
    confirmers = []
    for s in g.ENSEMBLE_SEEDS:
        m = SNNClassifier(in_features=inf, hidden=CONF_H, n_classes=5, n_rr=3).to(device)
        m.load_state_dict(torch.load(f"models/neurobeat-veb-v1/confirmer_seed{s}.pt",
                                     map_location=device))
        confirmers.append(m)

    def cascade_pred(windows, rr_raw):
        rr = torch.from_numpy(((rr_raw - mu) / sd).astype(np.float32)).to(device)
        xs = torch.from_numpy(np.stack([encode_beat(w, SCR_THR, SCR_T, ORDERS)
                                        for w in windows]).astype(np.float32)).to(device)
        xc = torch.from_numpy(np.stack([encode_beat(w, CONF_THR, CONF_T, ORDERS)
                                        for w in windows]).astype(np.float32)).to(device)
        scr = L.logits_of(screener, xs, rr, 512)
        conf = np.mean([L.logits_of(m, xc, rr, 512) for m in confirmers], axis=0)
        flag = (scr + b1).argmax(1) == VEB
        return flag & ((conf + b2).argmax(1) == VEB)

    def detect(sig_raw, sig_filt, which):
        if which == "xqrs":
            try:
                return np.asarray(xqrs_detect(sig=sig_raw, fs=cfg.data.fs, verbose=False))
            except Exception:
                return np.array([], dtype=int)
        return find_r_peaks(sig_filt, fs=cfg.data.fs)

    DETECTORS = ["xqrs", "light"]
    tot = {d: dict(true_beats=0, true_veb=0, det=0, det_tp=0, det_fp=0,
                   e2e_tp=0, e2e_fp=0, veb_missed_qrs=0, veb_missed_cls=0) for d in DETECTORS}

    for rid in DS2_RECORDS:
        rec = load_record(cfg.data.data_dir, rid, cfg.data.lead_index)
        sig = normalize(bandpass_filter(rec.signal, fs=rec.fs, low=cfg.data.bandpass_low,
                                        high=cfg.data.bandpass_high, order=cfg.data.filter_order))
        n = len(sig)

        # true beats with valid AAMI class and valid window (matches the locked DS2 set)
        ts, tl = [], []
        for samp, sym in zip(rec.ann_samples, rec.ann_symbols):
            cls = symbol_to_aami(sym)
            if cls is None or samp - wb < 0 or samp + wa > n:
                continue
            ts.append(int(samp)); tl.append(_CLASS_INDEX[cls])
        ts, tl = np.array(ts), np.array(tl)

        line = [rid]
        for which in DETECTORS:
            det = detect(rec.signal, sig, which)
            det = np.array([int(p) for p in sorted(det) if p - wb >= 0 and p + wa <= n])
            if len(det) == 0:
                continue
            det_to_true, used = greedy_match(det, ts, tol)

            diffs = np.diff(det.astype(np.float64))
            med = float(np.median(diffs)) if len(diffs) else 1.0
            med = med if med > 0 else 1.0
            pre = np.full(len(det), med); post = np.full(len(det), med)
            if len(diffs):
                pre[1:] = diffs; post[:-1] = diffs
            rr_raw = np.stack([pre / med, post / med, pre / np.maximum(post, 1.0)], axis=1)
            windows = [sig[p - wb:p + wa] for p in det]
            veb_pred = cascade_pred(windows, rr_raw)

            true_veb_idx = set(np.where(tl == VEB)[0].tolist())
            e2e_tp = e2e_fp = 0
            matched_pred = {}
            for i in range(len(det)):
                tj = det_to_true[i]
                if veb_pred[i]:
                    if tj >= 0 and tl[tj] == VEB:
                        e2e_tp += 1
                    else:
                        e2e_fp += 1
                if tj >= 0:
                    matched_pred[tj] = matched_pred.get(tj, False) or bool(veb_pred[i])
            missed_qrs = sum(1 for k in true_veb_idx if not used[k])
            missed_cls = sum(1 for k in true_veb_idx if used[k] and not matched_pred.get(k, False))
            nvt = int((tl == VEB).sum())
            det_tp = int(used.sum()); det_fp = int((det_to_true == -1).sum())
            t = tot[which]
            t["true_beats"] += len(ts); t["true_veb"] += nvt
            t["det"] += len(det); t["det_tp"] += det_tp; t["det_fp"] += det_fp
            t["e2e_tp"] += e2e_tp; t["e2e_fp"] += e2e_fp
            t["veb_missed_qrs"] += missed_qrs; t["veb_missed_cls"] += missed_cls
            line.append(f"{which}: detS {det_tp/max(len(ts),1):.3f} "
                        f"VEB {e2e_tp/max(nvt,1):.3f}/{e2e_tp/max(e2e_tp+e2e_fp,1):.3f}")
        print("  ".join(line), flush=True)

    out = {"tolerance_ms": int(TOL_S * 1000),
           "annotation_based_reference": {"sensitivity": 0.923, "ppv": 0.616}}
    print("\n=== DS2 end-to-end (detected R-peaks) ===", flush=True)
    for which in DETECTORS:
        t = tot[which]
        det_sens = t["det_tp"] / max(t["true_beats"], 1)
        det_ppv = t["det_tp"] / max(t["det_tp"] + t["det_fp"], 1)
        e2e_sens = t["e2e_tp"] / max(t["true_veb"], 1)
        e2e_ppv = t["e2e_tp"] / max(t["e2e_tp"] + t["e2e_fp"], 1)
        out[which] = {"detector_beat_sensitivity": round(det_sens, 4),
                      "detector_beat_ppv": round(det_ppv, 4),
                      "false_detections": t["det_fp"], "true_beats": t["true_beats"],
                      "end_to_end_VEB_sensitivity": round(e2e_sens, 4),
                      "end_to_end_VEB_ppv": round(e2e_ppv, 4),
                      "true_veb": t["true_veb"], "veb_missed_by_qrs": t["veb_missed_qrs"],
                      "veb_missed_by_classifier": t["veb_missed_cls"]}
        print(f"[{which:5}] detector beat sens {det_sens:.4f} ppv {det_ppv:.4f} "
              f"({t['det_fp']} false det) | end-to-end VEB {e2e_sens:.4f}/{e2e_ppv:.4f} "
              f"| VEB missed: {t['veb_missed_qrs']} by QRS, {t['veb_missed_cls']} by classifier",
              flush=True)
    print("(annotation-based reference: 0.923 / 0.616)", flush=True)
    Path("runs/end2end_rpeak.json").write_text(json.dumps(out, indent=2))
    print("\nsaved -> runs/end2end_rpeak.json", flush=True)


if __name__ == "__main__":
    main()
