"""Two-stage VEB cascade: cheap high-recall screener + gated high-precision confirmer.

Stage 1 (screener): a cheaper, independent SNN at half the timesteps (T32), tuned
  on DS1-val for high VEB recall (~0.97). Runs on EVERY beat; emits candidates.
Stage 2 (confirmer): the frozen NeuroBeat-VEB v1 (T64), high-precision operating
  point, run ONLY on Stage-1 candidates. A beat is VEB iff BOTH stages fire.

Stage 1 must be a *different* model from v1 (different timesteps -> different
encoding -> partly independent errors), otherwise re-judging its own candidates
adds nothing and the cascade collapses to a single threshold.

Energy: average SynOps/beat = SynOps(stage1, every beat)
                            + flag_rate * SynOps(stage2, flagged beats).
Because Stage 2 runs only on the small candidate set, its high-precision (heavier)
judgment costs little on average. Both operating points are fit on val only.

Requires models/neurobeat-veb-v1/weights.pt (run freeze_veb_v1.py first).

Usage:
  python experiments/two_stage_veb.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from freeze_veb_v1 import V1_CONFIG, EXTERNALS, ART, VEB  # noqa: E402
from lock_snn_rr import (  # noqa: E402
    N_CLASSES,
    N_RR,
    class_weights,
    load_data,
    logits_of,
    synops_per_beat,
    train_model,
)
from neurocardio.config import Config  # noqa: E402
from neurocardio.models.snn import SNNClassifier  # noqa: E402
from neurocardio.train.loop import resolve_device  # noqa: E402

# Cheap, independent screener: half the timesteps of v1, same everything else.
STAGE1_CONFIG = {
    "threshold": 0.12,
    "n_timesteps": 32,
    "orders": [0, 1],
    "hidden": 128,
    "scheme": "sqrt",
    "lr": 0.004,
    "epochs": 100,
    "batch": 512,
}
STAGE1_RECALL = 0.97   # screener must not drop VEBs the confirmer can never recover
SENS_TARGET = 0.90
SENS_BUFFER = 0.03
GRID = np.linspace(-3, 9, 49)


def _bin(pred_bool, true_bool):
    tp = int((pred_bool & true_bool).sum())
    fn = int((~pred_bool & true_bool).sum())
    fp = int((pred_bool & ~true_bool).sum())
    return tp / max(tp + fn, 1), tp / max(tp + fp, 1)


def veb_flags(logits, bias):
    return (logits + bias).argmax(1) == VEB


def fit_screener(logits, labels, recall_target=STAGE1_RECALL):
    """Stage-1 bias: VEB recall >= target, then maximize PPV to keep the flag
    rate (and thus Stage-2 cost) as low as possible."""
    true = labels == VEB
    best, best_effort = None, None
    for bS in GRID:
        for bV in GRID:
            bias = np.array([0.0, bS, bV, -12.0, -12.0])
            vs, vp = _bin(veb_flags(logits, bias), true)
            if vs >= recall_target and (best is None or vp > best[1]):
                best = (bias, vp)
            if best_effort is None or vs > best_effort[1]:
                best_effort = (bias, vs)
    return (best[0], True) if best else (best_effort[0], False)


def fit_confirmer(m2_logits, flag_mask, labels, sens_target=SENS_TARGET, buffer=SENS_BUFFER):
    """Stage-2 bias fit on the *cascade*: a beat is VEB iff Stage 1 flagged it AND
    Stage 2 confirms. Maximize cascade VEB PPV subject to cascade VEB sens >= target."""
    true = labels == VEB
    tgt = sens_target + buffer
    best, best_effort = None, None
    for bS in GRID:
        for bV in GRID:
            bias = np.array([0.0, bS, bV, -12.0, -12.0])
            casc = flag_mask & veb_flags(m2_logits, bias)
            vs, vp = _bin(casc, true)
            if vs >= tgt and (best is None or vp > best[1]):
                best = (bias, vp)
            if best_effort is None or vs > best_effort[1]:
                best_effort = (bias, vs)
    return (best[0], True) if best else (best_effort[0], False)


def cascade_eval(m1_logits, m2_logits, labels, bias1, bias2):
    flag = veb_flags(m1_logits, bias1)
    casc = flag & veb_flags(m2_logits, bias2)
    vs, vp = _bin(casc, labels == VEB)
    return {"VEB_sens": round(vs, 4), "VEB_ppv": round(vp, 4),
            "flag_rate": round(float(flag.mean()), 4), "n": int(len(labels))}


def main():
    device = resolve_device("auto")
    cfg = Config()
    w = ART / "weights.pt"
    if not w.exists():
        sys.exit(f"missing {w} - run experiments/freeze_veb_v1.py first")

    # Stage 2 = frozen v1 (T64). Load exact weights; do not retrain.
    m2 = SNNClassifier(in_features=2 * len(V1_CONFIG["orders"]), hidden=V1_CONFIG["hidden"],
                       n_classes=N_CLASSES, n_rr=N_RR).to(device)
    m2.load_state_dict(torch.load(w, map_location=device))
    m2.eval()

    # Encodings: T64 for the confirmer, T32 for the screener. Labels/RR identical.
    print("encoding T64 (confirmer) + T32 (screener)...", flush=True)
    d64 = load_data(cfg, V1_CONFIG, device, external_specs=EXTERNALS, augment_specs=None)
    d32 = load_data(cfg, STAGE1_CONFIG, device, external_specs=EXTERNALS, augment_specs=None)
    trx32, trr, trl, vax32, var, val_l, tex32, ter, tel, ext32 = d32
    _, _, _, vax64, _, _, tex64, _, _, ext64 = d64
    batch = 512

    # Train the Stage-1 screener on pure DS1-train.
    t0 = time.time()
    weight = class_weights(trl.cpu().numpy(), STAGE1_CONFIG["scheme"])
    m1 = train_model(trx32, trr, trl, STAGE1_CONFIG["hidden"], 2 * len(STAGE1_CONFIG["orders"]),
                     weight, STAGE1_CONFIG["lr"], STAGE1_CONFIG["epochs"], batch, device, 0)
    print(f"trained screener ({time.time()-t0:.0f}s)", flush=True)

    # ---- fit both operating points on val only ----
    m1_val = logits_of(m1, vax32, var, batch)
    bias1, ok1 = fit_screener(m1_val, val_l)
    flag_val = veb_flags(m1_val, bias1)
    m2_val = logits_of(m2, vax64, var, batch)
    bias2, ok2 = fit_confirmer(m2_val, flag_val, val_l)
    print(f"screener recall-feasible={ok1} bias1={[round(float(x),2) for x in bias1]}", flush=True)
    print(f"confirmer sens-feasible={ok2} bias2={[round(float(x),2) for x in bias2]}", flush=True)

    # ---- evaluate the frozen cascade on DS2 + externals ----
    def eval_db(x32, x64, rr_, lab):
        return cascade_eval(logits_of(m1, x32, rr_, batch),
                            logits_of(m2, x64, rr_, batch), lab, bias1, bias2)

    ds2 = eval_db(tex32, tex64, ter, tel)
    results = {"DS2": ds2}
    for n in ext32:
        ex32, er, el = ext32[n]
        ex64, _, _ = ext64[n]
        results[n] = eval_db(ex32, ex64, er, el)

    # ---- average SynOps: screener on every beat + confirmer on flagged beats ----
    syn1 = synops_per_beat(m1, tex32[:2048])
    flag_ds2 = veb_flags(logits_of(m1, tex32, ter, batch), bias1)
    fidx = np.where(flag_ds2)[0]
    syn2 = synops_per_beat(m2, tex64[fidx[:2048]]) if len(fidx) else 0.0
    flag_rate = float(flag_ds2.mean())
    avg_syn = syn1 + flag_rate * syn2

    print("\n=== two-stage cascade (frozen, val-locked) ===", flush=True)
    for db, m in results.items():
        print(f"  {db:7s} VEB sens {m['VEB_sens']:.3f}  PPV {m['VEB_ppv']:.3f}  "
              f"(flag_rate {m['flag_rate']:.3f}, n={m['n']})", flush=True)
    print(f"\n  SynOps: screener {syn1:.0f}/beat + flag_rate {flag_rate:.3f} x "
          f"confirmer {syn2:.0f} = {avg_syn:.0f} avg/beat  (budget 25000)", flush=True)

    out = {
        "model": "NeuroBeat-VEB v1 two-stage cascade",
        "stage1": {"role": "high-recall screener", **{k: STAGE1_CONFIG[k] for k in
                   ["threshold", "n_timesteps", "orders", "hidden"]},
                   "bias": [round(float(x), 3) for x in bias1],
                   "recall_target": STAGE1_RECALL, "recall_feasible": bool(ok1)},
        "stage2": {"role": "high-precision confirmer (frozen v1)",
                   **{k: V1_CONFIG[k] for k in ["threshold", "n_timesteps", "orders", "hidden"]},
                   "bias": [round(float(x), 3) for x in bias2], "sens_feasible": bool(ok2)},
        "results": results,
        "synops": {"stage1_per_beat": round(syn1, 1), "stage2_per_flagged": round(syn2, 1),
                   "flag_rate": round(flag_rate, 4), "average_per_beat": round(avg_syn, 1),
                   "budget": 25000},
        "protocol": "both operating points fit on DS1-val only; DS2/SVDB/INCART frozen",
    }
    (ART / "cascade.json").write_text(json.dumps(out, indent=2))
    torch.save({k: v.cpu() for k, v in m1.state_dict().items()}, ART / "stage1_weights.pt")
    print(f"\nsaved -> {ART}/cascade.json + stage1_weights.pt", flush=True)


if __name__ == "__main__":
    main()
