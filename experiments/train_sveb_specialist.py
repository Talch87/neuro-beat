"""NeuroBeat-SVEB specialist: an SVEB-first model, kept separate from VEB v1.

Where VEB v1 optimizes ventricular detection, this model inverts every choice
toward the supraventricular class:
  - Objective: SVEB-vs-rest. Operating point maximizes SVEB sensitivity subject
    to a PPV floor (SVEB PPV is inherently low - it is not the headline metric).
  - Training: SVEB-rich. svdb (supraventricular-dense) and INCART supraventricular
    beats are ADDED to DS1-train. DS2 stays the untouched test.
  - Higher time resolution (T96) - the sweep showed it is the best SVEB lever.

DS2 is never in training and its operating point is never tuned on it, so the
reported DS2 SVEB number is honest. svdb/INCART are training data here, so they
are NOT clean external tests for this model.

Usage:
  python experiments/train_sveb_specialist.py --seeds 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lock_snn_rr import (  # noqa: E402
    N_CLASSES,
    class_weights,
    eval_split,
    load_data,
    logits_of,
    synops_per_beat,
    train_model,
    _metrics,
)
from neurocardio.config import Config  # noqa: E402
from neurocardio.train.loop import resolve_device  # noqa: E402

SVEB_CONFIG = {
    "threshold": 0.12,
    "n_timesteps": 96,          # best SVEB lever from the RunPod sweep
    "orders": [0, 1],
    "hidden": 128,
    "scheme": "sqrt",
    "lr": 0.004,
    "epochs": 100,
    "batch": 512,
}
SVEB = 1  # class index
PPV_FLOOR = 0.20   # keep the detector from flagging everything; SVEB PPV is not the target
AUGMENT = [("svdb", "data/svdb", 0), ("incart", "data/incartdb", 1)]
ART = Path("models/neurobeat-sveb-specialist")
GRID = np.linspace(-3, 9, 49)


def fit_bias_sveb(logits, labels, ppv_floor=PPV_FLOOR):
    """SVEB-only operating point: maximize SVEB sensitivity subject to SVEB PPV
    >= ppv_floor on this (validation) split. VEB is not a target here."""
    best, best_effort = None, None
    for bV in GRID:
        for bS in GRID:
            bias = np.array([0.0, bS, bV, -12.0, -12.0])
            pred = (logits + bias).argmax(1)
            ss, sp = _metrics(pred, labels, SVEB)
            if sp >= ppv_floor and (best is None or ss > best[1]):
                best = (bias, ss)
            if best_effort is None or ss > best_effort[1]:
                best_effort = (bias, ss)
    return (best[0], True) if best else (best_effort[0], False)


def clean(rec):
    return {k: v for k, v in rec.items() if k != "state"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/sveb_specialist.json")
    args = ap.parse_args()
    device = resolve_device("auto")
    cfg = Config()

    print(f"device={device} config={SVEB_CONFIG} augment={[a[0] for a in AUGMENT]}", flush=True)
    # svdb + INCART added to TRAINING; DS2 is the built-in untouched test.
    data = load_data(cfg, SVEB_CONFIG, device, external_specs=None, augment_specs=AUGMENT)
    trx, trr, trl, vax, var, val_l, tex, ter, tel, _ = data
    weight = class_weights(trl.cpu().numpy(), SVEB_CONFIG["scheme"])
    batch = SVEB_CONFIG["batch"]
    in_features = 2 * len(SVEB_CONFIG["orders"])

    seeds = []
    for s in range(args.seeds):
        t0 = time.time()
        model = train_model(trx, trr, trl, SVEB_CONFIG["hidden"], in_features, weight,
                            SVEB_CONFIG["lr"], SVEB_CONFIG["epochs"], batch, device, s)
        bias, feasible = fit_bias_sveb(logits_of(model, vax, var, batch), val_l)
        val_m = eval_split(model, vax, var, val_l, bias, batch)
        ds2 = eval_split(model, tex, ter, tel, bias, batch)
        synops = synops_per_beat(model, tex[:2048])
        rec = {"seed": s, "bias": [round(float(x), 3) for x in bias],
               "val_feasible": bool(feasible), "val": val_m, "ds2": ds2,
               "synops": round(synops, 1), "seconds": round(time.time() - t0, 1),
               "state": {k: v.cpu() for k, v in model.state_dict().items()}}
        seeds.append(rec)
        print(f"seed {s}: VAL SVEB {val_m['SVEB_sens']}/{val_m['SVEB_ppv']} | "
              f"DS2 SVEB {ds2['SVEB_sens']}/{ds2['SVEB_ppv']} | "
              f"DS2 VEB {ds2['VEB_sens']}/{ds2['VEB_ppv']} | syn={synops:.0f} "
              f"({rec['seconds']}s)", flush=True)

    pool = [r for r in seeds if r["val_feasible"]] or seeds
    chosen = max(pool, key=lambda r: r["val"]["SVEB_sens"])
    sveb = np.array([r["ds2"]["SVEB_sens"] for r in seeds], dtype=float)
    print(f"\nDS2 SVEB sens over {args.seeds} seeds: mean {sveb.mean():.4f} min {sveb.min():.4f}",
          flush=True)
    print(f"chosen seed {chosen['seed']} (val SVEB sens {chosen['val']['SVEB_sens']})", flush=True)

    ART.mkdir(parents=True, exist_ok=True)
    torch.save(chosen["state"], ART / "weights.pt")
    (ART / "operating_point.json").write_text(json.dumps({
        "model": "NeuroBeat-SVEB specialist",
        "task": "SVEB-vs-rest (supraventricular ectopic beat detection)",
        "architecture": {k: SVEB_CONFIG[k] for k in ["threshold", "n_timesteps", "orders", "hidden"]},
        "bias": chosen["bias"], "chosen_seed": chosen["seed"],
        "training_data": "MIT-BIH DS1-train + svdb + INCART supraventricular (Normal capped)",
        "operating_point": f"SVEB-only: max SVEB sens s.t. SVEB PPV >= {PPV_FLOOR} on DS1-val",
        "val": chosen["val"], "ds2": chosen["ds2"], "synops_per_beat": chosen["synops"],
        "ds2_sveb_sens_over_seeds": {"seeds": args.seeds,
                                     "mean": round(float(sveb.mean()), 4),
                                     "min": round(float(sveb.min()), 4)},
    }, indent=2))
    Path(args.out).write_text(json.dumps([clean(r) for r in seeds], indent=2))
    print(f"\nsaved -> {ART}/weights.pt + operating_point.json", flush=True)


if __name__ == "__main__":
    main()
