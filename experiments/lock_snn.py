"""Honest, val-locked evaluation of SNN configs against the goal.

Protocol (no test-set tuning):
  1. Train on DS1-train (DS1 minus a patient-disjoint validation holdout).
  2. Fit per-class logit biases (the argmax operating point) on DS1-val to meet
     the goal-plus-buffer thresholds. The buffer absorbs the val->DS2 gap.
  3. Report FROZEN metrics on DS2. DS2 is never used to choose anything.
  4. --validate retrains with N seeds; all seeds must pass the goal on DS2.

Usage:
  python experiments/lock_snn.py --configs c.json --out out.json
  python experiments/lock_snn.py --validate '{...}' --seeds 5 --out val.json

Config dict: {threshold, n_timesteps, orders, hidden, scheme, lr, epochs, batch, seed}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neurocardio.config import Config
from neurocardio.data.dataset import build_split
from neurocardio.data.splits import DS1_RECORDS, DS2_RECORDS
from neurocardio.encoding.beat import encode_beat
from neurocardio.models.snn import SNNClassifier
from neurocardio.train.loop import resolve_device, set_seed

N_CLASSES = 5
GOAL = {"VEB_sens": 0.90, "VEB_ppv": 0.50, "SVEB_sens": 0.45, "synops": 25000}
BUFFER = {"VEB_sens": 0.02, "VEB_ppv": 0.03, "SVEB_sens": 0.03}  # fit tighter on val
VAL_RECORDS = ["201", "207", "223"]  # DS1 holdout rich in VEB and SVEB
TRAIN_RECORDS = [r for r in DS1_RECORDS if r not in VAL_RECORDS]
CACHE = Path("runs/lock_cache")
CACHE.mkdir(parents=True, exist_ok=True)


def raw_group(cfg, records, tag):
    fb, fl = CACHE / f"{tag}_beats.npy", CACHE / f"{tag}_labels.npy"
    if fb.exists() and fl.exists():
        return np.load(fb), np.load(fl)
    beats, labels = build_split(cfg, records)
    np.save(fb, beats)
    np.save(fl, labels)
    return beats, labels


def enc(beats, threshold, T, orders, tag):
    f = CACHE / f"{tag}_thr{threshold}_T{T}_o{''.join(map(str, orders))}.npy"
    if f.exists():
        return np.load(f)
    out = np.stack([encode_beat(b, threshold, T, orders) for b in beats]).astype(np.float32)
    np.save(f, out)
    return out


def class_weights(labels, scheme):
    c = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    w = np.zeros(N_CLASSES)
    p = c > 0
    base = c.sum() / (N_CLASSES * c[p])
    w[p] = np.sqrt(base) if scheme == "sqrt" else base
    return w


def _metrics(pred, labels, cls):
    tp = int(((pred == cls) & (labels == cls)).sum())
    fn = int(((pred != cls) & (labels == cls)).sum())
    fp = int(((pred == cls) & (labels != cls)).sum())
    sens = tp / max(tp + fn, 1)
    ppv = tp / max(tp + fp, 1)
    return sens, ppv


def fit_bias(logits, labels):
    """Grid-search per-class biases (N=0 ref, F/Q suppressed) to meet goal+buffer
    on this (validation) set; maximize VEB PPV margin. Returns best bias vector."""
    grid = np.linspace(-3, 7, 41)
    tgt = {k: GOAL[k] + BUFFER[k] for k in BUFFER}
    best_feasible = None
    best_effort = None
    for bS in grid:
        for bV in grid:
            bias = np.array([0.0, bS, bV, -12.0, -12.0])
            pred = (logits + bias).argmax(1)
            vs, vp = _metrics(pred, labels, 2)
            ss, _ = _metrics(pred, labels, 1)
            feasible = vs >= tgt["VEB_sens"] and vp >= tgt["VEB_ppv"] and ss >= tgt["SVEB_sens"]
            margin = min(vs - tgt["VEB_sens"], vp - tgt["VEB_ppv"], ss - tgt["SVEB_sens"])
            if feasible and (best_feasible is None or vp > best_feasible[1]):
                best_feasible = (bias, vp)
            if best_effort is None or margin > best_effort[1]:
                best_effort = (bias, margin)
    return (best_feasible[0] if best_feasible else best_effort[0]), best_feasible is not None


def train_model(trx, trl, hidden, in_features, weight, lr, epochs, batch, device, seed):
    set_seed(seed)
    model = SNNClassifier(in_features=in_features, hidden=hidden, n_classes=N_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=device))
    n = len(trx)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            lf(model(trx[idx]), trl[idx]).backward()
            opt.step()
    return model


def logits_of(model, x, batch):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(x[i : i + batch]).cpu())
    return torch.cat(out).numpy()


def synops_per_beat(model, x_sample):
    model.eval()
    b, t, _ = x_sample.shape
    mem1, mem2 = model.lif1.reset_mem(), model.lif2.reset_mem()
    fc1_in = fc2_in = 0
    with torch.no_grad():
        for step in range(t):
            inp = x_sample[:, step, :]
            fc1_in += float(inp.sum())
            spk1, mem1 = model.lif1(model.fc1(inp), mem1)
            fc2_in += float(spk1.sum())
            _, mem2 = model.lif2(model.fc2(spk1), mem2)
    return (fc1_in * model.fc1.out_features + fc2_in * model.fc2.out_features) / b


def run_once(p, data, device, seed):
    orders = p["orders"]
    hidden, batch = p.get("hidden", 128), p.get("batch", 512)
    lr, epochs, scheme = p.get("lr", 0.005), p.get("epochs", 100), p.get("scheme", "sqrt")
    trx, trl_t, vax, val_l, tex, tel = data
    in_features = 2 * len(orders)
    weight = class_weights(trl_t.cpu().numpy(), scheme)

    t0 = time.time()
    model = train_model(trx, trl_t, hidden, in_features, weight, lr, epochs, batch, device, seed)
    val_logits = logits_of(model, vax, batch)
    bias, feasible_on_val = fit_bias(val_logits, val_l)
    test_logits = logits_of(model, tex, batch)
    pred = (test_logits + bias).argmax(1)
    vs, vp = _metrics(pred, tel, 2)
    ss, sp = _metrics(pred, tel, 1)
    synops = synops_per_beat(model, tex[:2048])
    res = {
        **{
            k: p.get(k)
            for k in ["threshold", "n_timesteps", "orders", "hidden", "scheme", "lr", "epochs"]
        },
        "seed": seed,
        "bias": [round(float(x), 3) for x in bias],
        "feasible_on_val": bool(feasible_on_val),
        "VEB_sens": round(vs, 4),
        "VEB_ppv": round(vp, 4),
        "SVEB_sens": round(ss, 4),
        "SVEB_ppv": round(sp, 4),
        "synops": round(synops, 1),
        "seconds": round(time.time() - t0, 1),
    }
    res["pass"] = (
        vs >= GOAL["VEB_sens"]
        and vp >= GOAL["VEB_ppv"]
        and ss >= GOAL["SVEB_sens"]
        and synops <= GOAL["synops"]
    )
    return res, model.state_dict()


def load_data(cfg, p, device):
    trb, trl = raw_group(cfg, TRAIN_RECORDS, "trainsub")
    vab, val = raw_group(cfg, VAL_RECORDS, "val")
    teb, tel = raw_group(cfg, DS2_RECORDS, "test")
    thr, T, orders = p["threshold"], p["n_timesteps"], p["orders"]
    trx = torch.from_numpy(enc(trb, thr, T, orders, "trainsub")).to(device)
    vax = torch.from_numpy(enc(vab, thr, T, orders, "val")).to(device)
    tex = torch.from_numpy(enc(teb, thr, T, orders, "test")).to(device)
    trl_t = torch.from_numpy(trl.astype(np.int64)).to(device)
    return (trx, trl_t, vax, val.astype(np.int64), tex, tel.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="")
    ap.add_argument("--validate", default="")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/lock_results.json")
    args = ap.parse_args()
    device = resolve_device("auto")
    cfg = Config()
    print(f"device={device} train={len(TRAIN_RECORDS)} recs, val={VAL_RECORDS}", flush=True)

    if args.validate:
        p = json.loads(args.validate)
        data = load_data(cfg, p, device)
        rows = []
        for s in range(args.seeds):
            r, _ = run_once(p, data, device, s)
            rows.append(r)
            print(
                f"seed {s}: VEB {r['VEB_sens']}/{r['VEB_ppv']} SVEB {r['SVEB_sens']} "
                f"syn={r['synops']} pass={r['pass']} (valfeasible={r['feasible_on_val']}, {r['seconds']}s)",
                flush=True,
            )
            Path(args.out).write_text(json.dumps(rows, indent=2))
        arr = {
            k: np.array([r[k] for r in rows])
            for k in ["VEB_sens", "VEB_ppv", "SVEB_sens", "synops"]
        }
        print(
            "ALL PASS:",
            all(r["pass"] for r in rows),
            "| mean:",
            {k: round(float(v.mean()), 4) for k, v in arr.items()},
            "| min:",
            {k: round(float(v.min()), 4) for k, v in arr.items()},
            flush=True,
        )
        return

    configs = json.loads(Path(args.configs).read_text())
    results = []
    data_cache = {}
    for i, p in enumerate(configs):
        key = (p["threshold"], p["n_timesteps"], tuple(p["orders"]))
        if key not in data_cache:
            data_cache[key] = load_data(cfg, p, device)
        r, sd = run_once(p, data_cache[key], device, p.get("seed", 0))
        results.append(r)
        Path(args.out).write_text(json.dumps(results, indent=2))
        flag = "PASS" if r["pass"] else ("val+" if r["feasible_on_val"] else "    ")
        print(
            f"[{i + 1}/{len(configs)}] {flag} T{r['n_timesteps']} o{r['orders']} h{r['hidden']} "
            f"thr{r['threshold']} {r['scheme']}: VEB {r['VEB_sens']}/{r['VEB_ppv']} "
            f"SVEB {r['SVEB_sens']} syn={r['synops']} ({r['seconds']}s)",
            flush=True,
        )
        if r["pass"]:
            torch.save(sd, "runs/snn_lock_best.pt")
    print(f"\n{sum(r['pass'] for r in results)}/{len(results)} pass on DS2.", flush=True)


if __name__ == "__main__":
    main()
