"""Gated-ensemble VEB cascade: sparse screener + gated 3-seed ensemble confirmer.

The 5-seed ensemble reaches the accuracy target (DS2 VEB ~0.92 sens / ~0.63 PPV,
cross-database sensitivity ~0.90) but running K models on every beat blows the
energy budget. Here a single *genuinely sparse* screener runs on every beat and
the cached K-seed ensemble confirmer runs ONLY on the flagged candidates, so the
average energy stays within budget while the confirmed beats get ensemble-grade
judgment.

The confirmer needs no training or inference: the K seed models' logits are
already cached by freeze_veb_v1.py, so the confirmer is just the mean of cached
logits indexed by the screener's flag mask. Only the screener is trained here.

Average SynOps/beat = SynOps(screener, every beat)
                    + flag_rate * K * SynOps(one confirmer, flagged beats).

Both operating points are fit on DS1-val only; DS2/SVDB/INCART are frozen.

Usage:
  python experiments/gated_ensemble_veb.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from freeze_veb_v1 import V1_CONFIG, EXTERNALS, ART, VEB, CACHE  # noqa: E402
from lock_snn_rr import (  # noqa: E402
    N_CLASSES,
    N_RR,
    class_weights,
    load_data,
    logits_of,
    synops_per_beat,
    train_model,
    _metrics,
)
from neurocardio.config import Config  # noqa: E402
from neurocardio.models.snn import SNNClassifier  # noqa: E402
from neurocardio.train.loop import resolve_device  # noqa: E402

# Genuinely sparse screener: fewer hidden units + higher threshold => fewer
# spikes => low SynOps (unlike a T32 model, which count-pooling keeps expensive).
SCREENER = {
    "threshold": 0.18, "n_timesteps": 32, "orders": [0, 1],
    "hidden": 64, "scheme": "sqrt", "lr": 0.004, "epochs": 100, "batch": 512,
}
ENSEMBLE_SEEDS = [0, 1, 2]     # cached confirmer members (best K=3 by offline test)
# Op-point fit targets, selected on val (max val PPV s.t. val cascade sens >= 0.90):
SCREENER_RECALL = 0.97
SENS_TARGET = 0.90
SENS_BUFFER = 0.0
GRID = np.linspace(-3, 9, 49)
SPLITS = ["val", "ds2", "svdb", "incart"]
OUT = ART / "gated_ensemble.json"


def _bin(pred, true):
    tp = int((pred & true).sum()); fn = int((~pred & true).sum()); fp = int((pred & ~true).sum())
    return tp / max(tp + fn, 1), tp / max(tp + fp, 1)


def veb_flags(logits, bias):
    return (logits + bias).argmax(1) == VEB


def fit_screener(logits, labels, recall_target=SCREENER_RECALL):
    true = labels == VEB
    best, effort = None, None
    for bS in GRID:
        for bV in GRID:
            bias = np.array([0.0, bS, bV, -12.0, -12.0])
            vs, vp = _bin(veb_flags(logits, bias), true)
            if vs >= recall_target and (best is None or vp > best[1]):
                best = (bias, vp)
            if effort is None or vs > effort[1]:
                effort = (bias, vs)
    return (best[0], True) if best else (effort[0], False)


def fit_confirmer(m2, flag, labels, tgt=SENS_TARGET + SENS_BUFFER):
    true = labels == VEB
    best, effort = None, None
    for bS in GRID:
        for bV in GRID:
            bias = np.array([0.0, bS, bV, -12.0, -12.0])
            casc = flag & veb_flags(m2, bias)
            vs, vp = _bin(casc, true)
            if vs >= tgt and (best is None or vp > best[1]):
                best = (bias, vp)
            if effort is None or vs > effort[1]:
                effort = (bias, vs)
    return (best[0], True) if best else (effort[0], False)


def cascade(m1, m2, labels, b1, b2):
    flag = veb_flags(m1, b1)
    vs, vp = _bin(flag & veb_flags(m2, b2), labels == VEB)
    return {"VEB_sens": round(vs, 4), "VEB_ppv": round(vp, 4),
            "flag_rate": round(float(flag.mean()), 4), "n": int(len(labels))}


def main():
    device = resolve_device("auto")
    cfg = Config()
    for s in ENSEMBLE_SEEDS:
        if not (CACHE / f"seed{s}_ds2.npy").exists():
            sys.exit(f"missing cached confirmer logits for seed {s} - run freeze_veb_v1.py")
    lab = {n: np.load(CACHE / f"labels_{n}.npy") for n in SPLITS}
    # Confirmer = mean of cached K-seed logits (no training / no inference).
    conf = {n: np.mean([np.load(CACHE / f"seed{s}_{n}.npy") for s in ENSEMBLE_SEEDS], axis=0)
            for n in SPLITS}

    # Train the sparse screener on pure DS1-train; encode at its own config.
    print(f"encoding screener {SCREENER['orders']} h{SCREENER['hidden']} thr{SCREENER['threshold']} "
          f"T{SCREENER['n_timesteps']} ...", flush=True)
    d = load_data(cfg, SCREENER, device, external_specs=EXTERNALS, augment_specs=None)
    trx, trr, trl, vax, var, val_l, tex, ter, tel, ext = d
    batch = SCREENER["batch"]
    swf = ART / "screener_weights.pt"
    m1 = SNNClassifier(in_features=2 * len(SCREENER["orders"]), hidden=SCREENER["hidden"],
                       n_classes=N_CLASSES, n_rr=N_RR).to(device)
    if swf.exists():
        m1.load_state_dict(torch.load(swf, map_location=device))
        print("loaded cached screener weights", flush=True)
    else:
        t0 = time.time()
        weight = class_weights(trl.cpu().numpy(), SCREENER["scheme"])
        m1 = train_model(trx, trr, trl, SCREENER["hidden"], 2 * len(SCREENER["orders"]),
                         weight, SCREENER["lr"], SCREENER["epochs"], batch, device, 0)
        print(f"trained screener ({time.time()-t0:.0f}s)", flush=True)

    scr = {"val": logits_of(m1, vax, var, batch), "ds2": logits_of(m1, tex, ter, batch)}
    for n, (ex, er, el) in ext.items():
        scr[n] = logits_of(m1, ex, er, batch)

    b1, ok1 = fit_screener(scr["val"], lab["val"])
    flag_val = veb_flags(scr["val"], b1)
    b2, ok2 = fit_confirmer(conf["val"], flag_val, lab["val"])
    print(f"screener recall-feasible={ok1}; confirmer sens-feasible={ok2}", flush=True)

    results = {d_: cascade(scr[d_], conf[d_], lab[d_], b1, b2) for d_ in ["ds2", "svdb", "incart"]}

    # Energy: screener on every beat + flagged fraction * K * one-confirmer cost on flagged.
    syn_scr = synops_per_beat(m1, tex[:2048])
    flag_ds2 = veb_flags(scr["ds2"], b1)
    fidx = np.where(flag_ds2)[0]
    # one confirmer's SynOps on the flagged (ectopic-heavy) beats, at T64
    d64 = load_data(cfg, V1_CONFIG, device, external_specs=None, augment_specs=None)
    tex64 = d64[6]
    m2ref = SNNClassifier(in_features=2 * len(V1_CONFIG["orders"]), hidden=V1_CONFIG["hidden"],
                          n_classes=N_CLASSES, n_rr=N_RR).to(device)
    m2ref.load_state_dict(torch.load(CACHE / f"seed{ENSEMBLE_SEEDS[0]}.pt", map_location=device))
    syn_conf_flagged = synops_per_beat(m2ref, tex64[fidx[:2048]]) if len(fidx) else 0.0
    K = len(ENSEMBLE_SEEDS)
    flag_rate = float(flag_ds2.mean())
    avg_syn = syn_scr + flag_rate * K * syn_conf_flagged

    print("\n=== gated-ensemble cascade (frozen, val-locked) ===", flush=True)
    for db, m in results.items():
        print(f"  {db:7s} VEB sens {m['VEB_sens']:.3f}  PPV {m['VEB_ppv']:.3f}  "
              f"(flag_rate {m['flag_rate']:.3f}, n={m['n']})", flush=True)
    print(f"\n  SynOps: screener {syn_scr:.0f} + flag_rate {flag_rate:.3f} x K={K} x "
          f"confirmer {syn_conf_flagged:.0f} = {avg_syn:.0f} avg/beat  (budget 25000)", flush=True)
    hit = (results["ds2"]["VEB_sens"] >= 0.90 and results["ds2"]["VEB_ppv"] >= 0.60
           and avg_syn <= 25000)
    print(f"  ALL THREE CONSTRAINTS (DS2 sens>=0.90, ppv>=0.60, synops<=25k): {hit}", flush=True)

    out = {
        "model": "NeuroBeat-VEB v1 gated-ensemble cascade",
        "screener": {**{k: SCREENER[k] for k in ["threshold", "n_timesteps", "orders", "hidden"]},
                     "bias": [round(float(x), 3) for x in b1], "recall_target": SCREENER_RECALL,
                     "recall_feasible": bool(ok1)},
        "confirmer": {"ensemble_seeds": ENSEMBLE_SEEDS, "arch": {k: V1_CONFIG[k] for k in
                      ["threshold", "n_timesteps", "orders", "hidden"]},
                      "bias": [round(float(x), 3) for x in b2], "sens_feasible": bool(ok2)},
        "results": results,
        "synops": {"screener_per_beat": round(syn_scr, 1),
                   "confirmer_per_flagged": round(syn_conf_flagged, 1), "K": K,
                   "flag_rate": round(flag_rate, 4), "average_per_beat": round(avg_syn, 1),
                   "budget": 25000, "within_budget": bool(avg_syn <= 25000)},
        "meets_all_constraints": bool(hit),
        "protocol": "screener + confirmer operating points fit on DS1-val only; DS2/SVDB/INCART frozen",
    }
    OUT.write_text(json.dumps(out, indent=2))
    torch.save({k: v.cpu() for k, v in m1.state_dict().items()}, ART / "screener_weights.pt")
    print(f"\nsaved -> {OUT} + screener_weights.pt", flush=True)


if __name__ == "__main__":
    main()
