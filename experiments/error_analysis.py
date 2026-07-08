"""Per-record robustness, error composition, and paired statistical comparison.

Everything here is computed from the frozen gated cascade (reconstructed from
cached logits + frozen biases); nothing is retrained or invented. Produces:
  - per-record DS2 stats (record, #VEB, sens, PPV, FP, FN)
  - VEB-vs-rest confusion + false-positive breakdown by true class, DS2/SVDB/INCART
  - single-stage 5-class AAMI confusion on DS2
  - paired record-level bootstrap: cascade vs single-stage, cascade vs full ensemble
  - figure paper/figures/fig6_perrecord.png

Usage:
  python experiments/error_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import freeze_veb_v1 as fv  # noqa: E402
import gated_ensemble_veb as g  # noqa: E402
from lock_snn_rr import load_data, logits_of  # noqa: E402
from neurocardio.config import Config  # noqa: E402
from neurocardio.data.dataset import build_split_rr  # noqa: E402
from neurocardio.data.splits import DS2_RECORDS  # noqa: E402
from neurocardio.models.snn import SNNClassifier  # noqa: E402
from neurocardio.train.loop import resolve_device  # noqa: E402

VEB = 2
AAMI = ["N", "SVEB", "VEB", "F", "Q"]


def f1sp(pred, true):
    tp = int((pred & true).sum()); fn = int((~pred & true).sum()); fp = int((pred & ~true).sum())
    s = tp / max(tp + fn, 1); p = tp / max(tp + fp, 1)
    return (2 * s * p / max(s + p, 1e-9)), s, p


def main():
    device = resolve_device("auto"); cfg = Config()
    gj = json.loads(Path("models/neurobeat-veb-v1/gated_ensemble.json").read_text())
    b1 = np.array(gj["screener"]["bias"]); b2 = np.array(gj["confirmer"]["bias"])

    # screener logits on DS2 + externals (T32/thr0.18 encodings cached from the cascade run)
    d = load_data(cfg, g.SCREENER, device, external_specs=g.EXTERNALS, augment_specs=None)
    tex, ter = d[6], d[7]; ext32 = d[9]
    m1 = SNNClassifier(in_features=2 * len(g.SCREENER["orders"]), hidden=g.SCREENER["hidden"],
                       n_classes=5, n_rr=3).to(device)
    m1.load_state_dict(torch.load("models/neurobeat-veb-v1/screener_weights.pt", map_location=device))
    scr = {"ds2": logits_of(m1, tex, ter, 512)}
    for n, (ex, er, el) in ext32.items():
        scr[n] = logits_of(m1, ex, er, 512)
    conf = {n: np.mean([np.load(fv.CACHE / f"seed{s}_{n}.npy") for s in g.ENSEMBLE_SEEDS], axis=0)
            for n in ["ds2", "svdb", "incart"]}
    labels = {n: np.load(fv.CACHE / f"labels_{n}.npy") for n in ["ds2", "svdb", "incart"]}

    def casc(n):
        flag = (scr[n] + b1).argmax(1) == VEB
        return flag & ((conf[n] + b2).argmax(1) == VEB)
    pred = {n: casc(n) for n in ["ds2", "svdb", "incart"]}

    # ---- per-record DS2 ----
    rid = []
    for rec in DS2_RECORDS:
        beats, _, _ = build_split_rr(cfg, [rec]); rid.extend([rec] * len(beats))
    rid = np.array(rid)
    lab = labels["ds2"]; true = lab == VEB; p = pred["ds2"]
    assert len(rid) == len(lab)
    per_rec = []
    for rec in DS2_RECORDS:
        m = rid == rec; t = true[m]; pp = p[m]
        tp = int((pp & t).sum()); fn = int((~pp & t).sum()); fp = int((pp & ~t).sum())
        nveb = int(t.sum())
        sens = round(tp / max(tp + fn, 1), 3) if nveb else None
        ppv = round(tp / (tp + fp), 3) if (tp + fp) > 0 else None
        per_rec.append({"record": rec, "n_veb": nveb, "TP": tp, "FN": fn, "FP": fp,
                        "sens": sens, "ppv": ppv})

    # ---- VEB-vs-rest confusion + FP breakdown by true class ----
    conf_tbl = {}
    for n in ["ds2", "svdb", "incart"]:
        lb = labels[n]; pr = pred[n]; tr = lb == VEB
        tp = int((pr & tr).sum()); fn = int((~pr & tr).sum())
        fp = int((pr & ~tr).sum()); tn = int((~pr & ~tr).sum())
        fp_by = {AAMI[c]: int(((pr) & (lb == c)).sum()) for c in [0, 1, 3, 4]}
        conf_tbl[n] = {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "fp_by_true_class": fp_by}

    # ---- single-stage 5-class AAMI confusion on DS2 ----
    op = json.loads(Path("models/neurobeat-veb-v1/operating_point.json").read_text())
    ssbias = np.array(op["bias"]); ss_seed = op["chosen_seed"]
    ss_logits = np.load(fv.CACHE / f"seed{ss_seed}_ds2.npy")
    ss_argmax = (ss_logits + ssbias).argmax(1)
    cm = np.zeros((5, 5), int)
    for t, q in zip(lab, ss_argmax):
        cm[t, q] += 1

    # ---- paired record-level bootstrap (cascade vs single-stage, vs full ensemble) ----
    ss_veb = ss_argmax == VEB
    val_l = np.load(fv.CACHE / "labels_val.npy")
    ens_ds2 = np.mean([np.load(fv.CACHE / f"seed{s}_ds2.npy") for s in range(5)], axis=0)
    ens_val = np.mean([np.load(fv.CACHE / f"seed{s}_val.npy") for s in range(5)], axis=0)
    (ensb, _, _), _ = fv.op_sens_first(ens_val, val_l)
    ens_veb = (ens_ds2 + ensb).argmax(1) == VEB
    by_rec = {r: np.where(rid == r)[0] for r in DS2_RECORDS}
    R = len(DS2_RECORDS)

    def paired(a, b, metric):
        rng = np.random.default_rng(0); diffs = []
        for _ in range(2000):
            ch = rng.integers(0, R, R)
            idx = np.concatenate([by_rec[DS2_RECORDS[c]] for c in ch])
            fa = f1sp(a[idx], true[idx]); fb = f1sp(b[idx], true[idx])
            mi = {"F1": 0, "sens": 1, "ppv": 2}[metric]
            diffs.append(fa[mi] - fb[mi])
        diffs = np.sort(np.array(diffs))
        return {"median": round(float(np.median(diffs)), 4),
                "ci": [round(float(diffs[50]), 4), round(float(diffs[1949]), 4)],
                "frac_cascade_better": round(float((diffs > 0).mean()), 3)}

    cmp = {
        "cascade_vs_single_stage": {m: paired(pred["ds2"], ss_veb, m) for m in ["F1", "sens", "ppv"]},
        "cascade_vs_full_ensemble": {m: paired(pred["ds2"], ens_veb, m) for m in ["F1", "sens", "ppv"]},
    }

    out = {"per_record_ds2": per_rec, "veb_vs_rest_confusion": conf_tbl,
           "single_stage_ds2_aami_confusion": {"labels": AAMI, "matrix": cm.tolist(),
                                               "seed": ss_seed},
           "paired_record_bootstrap": cmp}
    Path("runs/error_analysis.json").write_text(json.dumps(out, indent=2))

    # ---- print a readable summary ----
    print("=== per-record DS2 (cascade) ===", flush=True)
    print("rec   nVEB  TP  FN  FP   sens   ppv", flush=True)
    for r in per_rec:
        print(f"{r['record']:>4} {r['n_veb']:>5} {r['TP']:>4} {r['FN']:>3} {r['FP']:>4}  "
              f"{r['sens'] if r['sens'] is not None else '  -':>5}  "
              f"{r['ppv'] if r['ppv'] is not None else '  -':>5}", flush=True)
    print("\n=== VEB-vs-rest confusion + FP source ===", flush=True)
    for n, c in conf_tbl.items():
        print(f"{n:6} TP {c['TP']:>5} FN {c['FN']:>4} FP {c['FP']:>5} TN {c['TN']:>6} | "
              f"FP from {c['fp_by_true_class']}", flush=True)
    print("\n=== single-stage DS2 AAMI confusion (rows=true, cols=pred; seed",
          ss_seed, ") ===", flush=True)
    print("        " + "  ".join(f"{a:>6}" for a in AAMI), flush=True)
    for i, a in enumerate(AAMI):
        print(f"{a:>5}  " + "  ".join(f"{cm[i,j]:>6}" for j in range(5)), flush=True)
    print("\n=== paired record-level bootstrap (F1 difference, cascade minus other) ===", flush=True)
    for k, v in cmp.items():
        print(f"{k}: F1 {v['F1']['median']:+.3f} CI {v['F1']['ci']} "
              f"(cascade better in {v['F1']['frac_cascade_better']*100:.0f}% of resamples)", flush=True)

    # ---- Figure 6: per-record sens & ppv ----
    recs = [r for r in per_rec if r["n_veb"] > 0]
    nveb = [r["n_veb"] for r in recs]
    sens = [r["sens"] for r in recs]
    ppv = [r["ppv"] if r["ppv"] is not None else 0 for r in recs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(nveb, sens, s=60, color="#1f77b4", edgecolor="black", label="VEB sensitivity", zorder=3)
    ax.scatter(nveb, ppv, s=60, color="#ff7f0e", marker="s", edgecolor="black", label="VEB PPV", zorder=3)
    ax.axhline(0.90, ls="--", color="#1f77b4", lw=1, alpha=0.6)
    ax.axhline(0.60, ls="--", color="#ff7f0e", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("VEB beats in record (log scale)")
    ax.set_ylabel("Per-record value")
    ax.set_title("Per-record VEB sensitivity and PPV on DS2 (gated cascade)")
    ax.set_ylim(0, 1.02); ax.legend(loc="lower right"); ax.grid(ls=":", alpha=0.4)
    fig.savefig("paper/figures/fig6_perrecord.png", dpi=200, bbox_inches="tight")
    fig.savefig("paper/figures/fig6_perrecord.pdf", bbox_inches="tight")
    plt.close(fig)
    print("\nsaved runs/error_analysis.json + paper/figures/fig6_perrecord.png", flush=True)


if __name__ == "__main__":
    main()
