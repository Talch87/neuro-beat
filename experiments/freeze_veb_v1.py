"""Freeze NeuroBeat-VEB v1: a VEB-only, val-locked SNN detector.

The VEB productization (steps 1, 2, 4). We stop compromising the VEB operating
point chasing SVEB and ship the VEB detector as a versioned artifact.

Design note: training is expensive and the operating point needs iteration, so
the two are decoupled. Phase 1 trains N seeds and caches, per seed, the weights
plus the raw logits on val / DS2 / SVDB / INCART. Phase 2 loads those logits and
fits operating points in milliseconds -- rerun with --refit-only to re-tune the
threshold without touching the GPU.

Honest protocol: every operating point is fit on DS1-val only, then frozen for
DS2 and the external databases. The frozen seed is selected by VAL performance.

Usage:
  python experiments/freeze_veb_v1.py --seeds 5          # train (cache-aware) + fit + freeze
  python experiments/freeze_veb_v1.py --refit-only       # re-fit from cache, no GPU
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
    load_data,
    logits_of,
    synops_per_beat,
    train_model,
    _metrics,
)
from neurocardio.config import Config  # noqa: E402
from neurocardio.models.snn import SNNClassifier  # noqa: E402
from neurocardio.train.loop import resolve_device  # noqa: E402

V1_CONFIG = {
    "threshold": 0.12, "n_timesteps": 64, "orders": [0, 1],
    "hidden": 128, "scheme": "sqrt", "lr": 0.004, "epochs": 100, "batch": 512,
}
VEB = 2
SENS_MIN = 0.90     # clinical floor: do not miss ventricular beats
PPV_MIN = 0.60      # product target band 0.60-0.70
SPLITS = ["val", "ds2", "svdb", "incart"]
EXTERNALS = [("svdb", "data/svdb", 0), ("incart", "data/incartdb", 1)]
CACHE = Path("runs/freeze_cache")
ART = Path("models/neurobeat-veb-v1")
GRID = np.linspace(-3, 9, 49)


# ------------------------- operating-point strategies -------------------------
def _search(logits, labels, feasible, score):
    """2-D (SVEB bias, VEB bias) search. Return (bias, sens, ppv) maximizing
    `score(sens, ppv)` over operating points where `feasible(sens, ppv)`; None
    if none are feasible."""
    best = None
    for bS in GRID:
        for bV in GRID:
            bias = np.array([0.0, bS, bV, -12.0, -12.0])
            pred = (logits + bias).argmax(1)
            vs, vp = _metrics(pred, labels, VEB)
            if feasible(vs, vp):
                sc = score(vs, vp)
                if best is None or sc > best[0]:
                    best = (sc, bias, vs, vp)
    return None if best is None else (best[1], best[2], best[3])


def op_sens_first(logits, labels, sens_min=SENS_MIN):
    """Max VEB PPV subject to VEB sens >= sens_min. If unreachable, take the
    highest achievable sensitivity and the best PPV there (never degenerate)."""
    hit = _search(logits, labels, lambda s, p: s >= sens_min, lambda s, p: p)
    if hit is not None:
        return hit, True
    smax = max(_metrics((logits + np.array([0.0, bS, bV, -12.0, -12.0])).argmax(1), labels, VEB)[0]
               for bS in GRID for bV in GRID)
    hit = _search(logits, labels, lambda s, p: s >= smax - 1e-9, lambda s, p: p)
    return hit, False


def op_ppv_first(logits, labels, ppv_min=PPV_MIN):
    """Max VEB sensitivity subject to VEB PPV >= ppv_min."""
    hit = _search(logits, labels, lambda s, p: p >= ppv_min, lambda s, p: s)
    return (hit, True) if hit is not None else (
        _search(logits, labels, lambda s, p: True, lambda s, p: p), False)


def op_balanced(logits, labels):
    """Max VEB F1."""
    hit = _search(logits, labels, lambda s, p: True,
                  lambda s, p: 2 * s * p / max(s + p, 1e-9))
    return hit, True


def frontier(logits, labels, sens_grid=(0.80, 0.85, 0.88, 0.90, 0.92, 0.95)):
    """1-D VEB-threshold sweep (SVEB natural): PPV achievable at each sens level."""
    rows = []
    sweep = []
    for bV in np.linspace(-3, 10, 66):
        pred = (logits + np.array([0.0, 0.0, bV, -12.0, -12.0])).argmax(1)
        sweep.append(_metrics(pred, labels, VEB))
    sweep = np.array(sweep)  # (n, 2): sens, ppv
    for st in sens_grid:
        ok = sweep[sweep[:, 0] >= st]
        rows.append((st, round(float(ok[:, 1].max()), 3) if len(ok) else float("nan")))
    return rows


def metr(logits, labels, bias):
    pred = (logits + bias).argmax(1)
    vs, vp = _metrics(pred, labels, VEB)
    return {"VEB_sens": round(vs, 4), "VEB_ppv": round(vp, 4), "n": int(len(labels))}


# ------------------------------- phases --------------------------------------
def train_and_cache(args):
    device = resolve_device("auto")
    cfg = Config()
    print(f"device={device} config={V1_CONFIG}", flush=True)
    data = load_data(cfg, V1_CONFIG, device, external_specs=EXTERNALS, augment_specs=None)
    trx, trr, trl, vax, var, val_l, tex, ter, tel, external = data
    batch = V1_CONFIG["batch"]
    weight = class_weights(trl.cpu().numpy(), V1_CONFIG["scheme"])
    in_features = 2 * len(V1_CONFIG["orders"])

    splits = {"val": (vax, var, val_l), "ds2": (tex, ter, tel)}
    for n, (ex, er, el) in external.items():
        splits[n] = (ex, er, el)
    CACHE.mkdir(parents=True, exist_ok=True)
    for n in SPLITS:
        np.save(CACHE / f"labels_{n}.npy", np.asarray(splits[n][2]))

    for s in range(args.seeds):
        wf = CACHE / f"seed{s}.pt"
        if wf.exists() and all((CACHE / f"seed{s}_{n}.npy").exists() for n in SPLITS):
            print(f"seed {s}: cached, skip", flush=True)
            continue
        t0 = time.time()
        model = train_model(trx, trr, trl, V1_CONFIG["hidden"], in_features, weight,
                            V1_CONFIG["lr"], V1_CONFIG["epochs"], batch, device, s)
        for n in SPLITS:
            x, rr, _ = splits[n]
            np.save(CACHE / f"seed{s}_{n}.npy", logits_of(model, x, rr, batch))
        syn = synops_per_beat(model, tex[:2048])
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, wf)
        (CACHE / f"seed{s}_meta.json").write_text(json.dumps({"synops": round(syn, 1)}))
        print(f"seed {s}: trained + cached, synops={syn:.0f} ({time.time()-t0:.0f}s)", flush=True)


def fit_and_freeze(args):
    labels = {n: np.load(CACHE / f"labels_{n}.npy") for n in SPLITS}
    seeds = sorted(int(p.stem[4:]) for p in CACHE.glob("seed*.pt"))
    if not seeds:
        sys.exit("no cached seeds - run without --refit-only first")

    print(f"\n=== VEB sensitivity-PPV frontier (DS2, per seed) ===", flush=True)
    print("      " + "  ".join(f"s>={st:.2f}" for st, _ in frontier(
        np.load(CACHE / f"seed{seeds[0]}_ds2.npy"), labels["ds2"])), flush=True)
    for s in seeds:
        fr = frontier(np.load(CACHE / f"seed{s}_ds2.npy"), labels["ds2"])
        print(f"seed{s} ppv " + "  ".join(f"{p:.3f} " for _, p in fr), flush=True)

    records = []
    for s in seeds:
        lg = {n: np.load(CACHE / f"seed{s}_{n}.npy") for n in SPLITS}
        syn = json.loads((CACHE / f"seed{s}_meta.json").read_text())["synops"]
        ops = {}
        for name, fn in [("sens_first", op_sens_first), ("ppv_first", op_ppv_first),
                         ("balanced", op_balanced)]:
            (bias, vs, vp), feas = fn(lg["val"], labels["val"])
            ops[name] = {
                "bias": [round(float(x), 3) for x in bias], "val_feasible": bool(feas),
                "val": {"VEB_sens": round(vs, 4), "VEB_ppv": round(vp, 4)},
                **{d: metr(lg[d], labels[d], bias) for d in ["ds2", "svdb", "incart"]},
            }
        records.append({"seed": s, "synops": syn, "ops": ops})

    for name in ["sens_first", "ppv_first", "balanced"]:
        print(f"\n=== operating point: {name} (frozen op point, per seed) ===", flush=True)
        for r in records:
            o = r["ops"][name]
            print(f"  seed{r['seed']}: val {o['val']['VEB_sens']}/{o['val']['VEB_ppv']} "
                  f"| DS2 {o['ds2']['VEB_sens']}/{o['ds2']['VEB_ppv']} "
                  f"| svdb {o['svdb']['VEB_sens']}/{o['svdb']['VEB_ppv']} "
                  f"| incart {o['incart']['VEB_sens']}/{o['incart']['VEB_ppv']} "
                  f"| syn {r['synops']:.0f} feas={o['val_feasible']}", flush=True)
        ds2s = np.array([r["ops"][name]["ds2"]["VEB_sens"] for r in records])
        ds2p = np.array([r["ops"][name]["ds2"]["VEB_ppv"] for r in records])
        print(f"  DS2 mean sens {ds2s.mean():.4f} (min {ds2s.min():.4f}) "
              f"ppv {ds2p.mean():.4f} (min {ds2p.min():.4f})", flush=True)

    # ---- freeze v1 at the clinically-sensible sens-first point ----
    # Choose among seeds that ACTUALLY reach the sensitivity target on val
    # (never an infeasible fallback), and break ties by val VEB F1 -- a general
    # "this model trained well" signal, not the degenerate high-PPV-at-low-sens point.
    def val_f1(r):
        v = r["ops"]["balanced"]["val"]
        s, p = v["VEB_sens"], v["VEB_ppv"]
        return 2 * s * p / max(s + p, 1e-9)

    freeze_op = "sens_first"
    feasible = [r for r in records if r["ops"][freeze_op]["val_feasible"]]
    chosen = max(feasible or records, key=val_f1)
    o = chosen["ops"][freeze_op]
    ART.mkdir(parents=True, exist_ok=True)
    torch.save(torch.load(CACHE / f"seed{chosen['seed']}.pt"), ART / "weights.pt")
    ds2s = np.array([r["ops"][freeze_op]["ds2"]["VEB_sens"] for r in records])
    ds2p = np.array([r["ops"][freeze_op]["ds2"]["VEB_ppv"] for r in records])
    op = {
        "model": "NeuroBeat-VEB v1",
        "task": "VEB-vs-rest (ventricular ectopic beat detection)",
        "architecture": {k: V1_CONFIG[k] for k in ["threshold", "n_timesteps", "orders", "hidden"]},
        "n_classes": N_CLASSES, "bias": o["bias"], "chosen_seed": chosen["seed"],
        "operating_point": f"sens-first: max VEB PPV s.t. VEB sens >= {SENS_MIN} on DS1-val",
        "training_data": "MIT-BIH DS1-train (pure, no augmentation)",
        "val": o["val"], "ds2": o["ds2"], "svdb": o["svdb"], "incart": o["incart"],
        "synops_per_beat": chosen["synops"],
        "ds2_over_seeds": {"seeds": len(records),
                           "VEB_sens": {"mean": round(float(ds2s.mean()), 4), "min": round(float(ds2s.min()), 4)},
                           "VEB_ppv": {"mean": round(float(ds2p.mean()), 4), "min": round(float(ds2p.min()), 4)}},
        "targets": {"VEB_sens": ">=0.90", "VEB_ppv": "0.60-0.70", "synops": "<=25000"},
    }
    (ART / "operating_point.json").write_text(json.dumps(op, indent=2))
    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\nfroze seed {chosen['seed']} @ {freeze_op} -> {ART}/", flush=True)
    print(f"  DS2 VEB {o['ds2']['VEB_sens']}/{o['ds2']['VEB_ppv']} synops {chosen['synops']:.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/freeze_veb_v1.json")
    ap.add_argument("--refit-only", action="store_true", help="skip training; re-fit from cache")
    args = ap.parse_args()
    if not args.refit_only:
        train_and_cache(args)
    fit_and_freeze(args)


if __name__ == "__main__":
    main()
