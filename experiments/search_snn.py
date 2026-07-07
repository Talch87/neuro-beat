"""Multi-objective search for a low-timestep SNN config meeting the goal:
VEB sens >= 0.90, VEB PPV >= 0.50, SVEB sens >= 0.45, SynOps/beat <= 25000,
inter-patient DS1->DS2, stable across seeds.

Usage:
  python experiments/search_snn.py --configs path/to/configs.json --out results.json
  python experiments/search_snn.py --validate '{"...params..."}' --seeds 5 --out val.json

Each config dict: {threshold, n_timesteps, orders, hidden, scheme, lr, epochs,
                   batch, seed, [weights]}
Encoded datasets are cached per (threshold, n_timesteps, orders) and kept on GPU.
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
from neurocardio.data.segment import AAMI_CLASSES
from neurocardio.data.splits import get_split
from neurocardio.encoding.beat import encode_beat
from neurocardio.eval.metrics import aami_metrics, confusion
from neurocardio.models.snn import SNNClassifier
from neurocardio.train.loop import resolve_device, set_seed

GOAL = {"VEB_sens": 0.90, "VEB_ppv": 0.50, "SVEB_sens": 0.45, "synops": 25000}
CACHE = Path("runs/search_cache")
CACHE.mkdir(parents=True, exist_ok=True)
N_CLASSES = 5


def raw_beats(cfg, split):
    fb, fl = CACHE / f"{split}_beats.npy", CACHE / f"{split}_labels.npy"
    if fb.exists() and fl.exists():
        return np.load(fb), np.load(fl)
    beats, labels = build_split(cfg, get_split(split))
    np.save(fb, beats)
    np.save(fl, labels)
    return beats, labels


def encode_cached(beats, threshold, n_timesteps, orders, tag):
    key = f"{tag}_thr{threshold}_T{n_timesteps}_o{''.join(map(str, orders))}.npy"
    f = CACHE / key
    if f.exists():
        return np.load(f)
    out = np.stack([encode_beat(b, threshold, n_timesteps, orders) for b in beats]).astype(
        np.float32
    )
    np.save(f, out)
    return out


def make_weights(labels, scheme, custom=None):
    if custom is not None:
        return np.asarray(custom, dtype=np.float64)
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    total = counts.sum()
    w = np.zeros(N_CLASSES, dtype=np.float64)
    present = counts > 0
    base = total / (N_CLASSES * counts[present])
    if scheme == "sqrt":
        w[present] = np.sqrt(base)
    elif scheme == "none":
        w[present] = 1.0
    else:  # balanced
        w[present] = base
    return w


def synops_per_beat(model, x_sample, device):
    """Mean synaptic operations per beat = presynaptic events x fan-out, summed
    over layers, averaged over the sample. Events are pooled spike counts."""
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
    synops = fc1_in * model.fc1.out_features + fc2_in * model.fc2.out_features
    return synops / b


def evaluate(model, x, y, batch, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            preds.append(model(x[i : i + batch]).argmax(1))
    cm = confusion(y.cpu().numpy(), torch.cat(preds).cpu().numpy(), N_CLASSES)
    return cm, aami_metrics(cm, AAMI_CLASSES)


def run_config(p, tr_b, tr_lab, te_b, te_lab, device):
    thr, T = p["threshold"], p["n_timesteps"]
    orders, hidden = p["orders"], p.get("hidden", 128)
    epochs, batch, lr = p.get("epochs", 50), p.get("batch", 512), p.get("lr", 0.005)
    seed = p.get("seed", 0)

    tr_x = torch.from_numpy(encode_cached(tr_b, thr, T, orders, "train")).to(device)
    te_x = torch.from_numpy(encode_cached(te_b, thr, T, orders, "test")).to(device)
    tr_y = torch.from_numpy(tr_lab.astype(np.int64)).to(device)
    te_y = torch.from_numpy(te_lab.astype(np.int64)).to(device)
    in_features = 2 * len(orders)

    set_seed(seed)
    model = SNNClassifier(in_features=in_features, hidden=hidden, n_classes=N_CLASSES).to(device)
    weight = torch.tensor(
        make_weights(tr_lab, p.get("scheme", "sqrt"), p.get("weights")),
        dtype=torch.float32,
        device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    n = len(tr_x)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss_fn(model(tr_x[idx]), tr_y[idx]).backward()
            opt.step()
    _, m = evaluate(model, te_x, te_y, batch, device)
    synops = synops_per_beat(model, te_x[:2048], device)
    veb, sveb = m["per_class"]["VEB"], m["per_class"]["SVEB"]
    res = {
        **{
            k: p.get(k)
            for k in [
                "threshold",
                "n_timesteps",
                "orders",
                "hidden",
                "scheme",
                "lr",
                "epochs",
                "seed",
            ]
        },
        "weights": p.get("weights"),
        "in_features": in_features,
        "VEB_sens": round(veb["sensitivity"], 4),
        "VEB_ppv": round(veb["ppv"], 4),
        "SVEB_sens": round(sveb["sensitivity"], 4),
        "SVEB_ppv": round(sveb["ppv"], 4),
        "acc": round(m["overall_accuracy"], 4),
        "synops": round(synops, 1),
        "seconds": round(time.time() - t0, 1),
    }
    res["pass"] = (
        res["VEB_sens"] >= GOAL["VEB_sens"]
        and res["VEB_ppv"] >= GOAL["VEB_ppv"]
        and res["SVEB_sens"] >= GOAL["SVEB_sens"]
        and res["synops"] <= GOAL["synops"]
    )
    return res, model.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="")
    ap.add_argument("--validate", default="")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/search_results.json")
    args = ap.parse_args()

    device = resolve_device("auto")
    cfg = Config()
    tr_b, tr_lab = raw_beats(cfg, "train")
    te_b, te_lab = raw_beats(cfg, "test")
    print(f"device={device} DS1={len(tr_lab)} DS2={len(te_lab)}", flush=True)

    if args.validate:
        base = json.loads(args.validate)
        rows = []
        for s in range(args.seeds):
            p = {**base, "seed": s}
            r, _ = run_config(p, tr_b, tr_lab, te_b, te_lab, device)
            rows.append(r)
            print(
                f"seed {s}: VEB sens={r['VEB_sens']} ppv={r['VEB_ppv']} "
                f"SVEB sens={r['SVEB_sens']} synops={r['synops']} pass={r['pass']} ({r['seconds']}s)",
                flush=True,
            )
        arr = {
            k: np.array([r[k] for r in rows])
            for k in ["VEB_sens", "VEB_ppv", "SVEB_sens", "synops"]
        }
        summary = {
            "config": base,
            "seeds": args.seeds,
            "all_pass": bool(all(r["pass"] for r in rows)),
            "mean": {k: round(float(v.mean()), 4) for k, v in arr.items()},
            "min": {k: round(float(v.min()), 4) for k, v in arr.items()},
            "rows": rows,
        }
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print("ALL SEEDS PASS:", summary["all_pass"], "| mean:", summary["mean"], flush=True)
        return

    configs = json.loads(Path(args.configs).read_text())
    results, best = [], None
    for i, p in enumerate(configs):
        r, sd = run_config(p, tr_b, tr_lab, te_b, te_lab, device)
        results.append(r)
        Path(args.out).write_text(json.dumps(results, indent=2))
        flag = "PASS" if r["pass"] else "    "
        print(
            f"[{i + 1}/{len(configs)}] {flag} T={r['n_timesteps']} o={r['orders']} h={r['hidden']} "
            f"thr={r['threshold']} {r['scheme']}: VEB {r['VEB_sens']}/{r['VEB_ppv']} "
            f"SVEB {r['SVEB_sens']} syn={r['synops']} ({r['seconds']}s)",
            flush=True,
        )
        if r["pass"] and (best is None or r["VEB_sens"] > best[0]):
            best = (r["VEB_sens"], sd)
            torch.save(sd, "runs/snn_search_best.pt")
    n_pass = sum(r["pass"] for r in results)
    print(f"\n{n_pass}/{len(results)} configs pass all constraints.", flush=True)


if __name__ == "__main__":
    main()
