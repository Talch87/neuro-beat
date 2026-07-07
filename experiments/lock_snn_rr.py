"""Honest, val-locked evaluation of RR-enabled SNN configs against the goal.

Same protocol as lock_snn.py, but the model also consumes the 3 patient-normalized
RR-interval features (see models.snn.SNNClassifier n_rr). The operating point (per-class
logit biases) is fit ONLY on the DS1 validation holdout and then frozen for DS2, so no
test-set tuning occurs.

Protocol:
  1. Train on DS1-train (DS1 minus a patient-disjoint validation holdout).
  2. Fit per-class logit biases on DS1-val to meet goal+buffer.
  3. Report FROZEN metrics on DS2. DS2 chooses nothing.
  4. --validate retrains with N seeds; report per-seed and aggregate.

Usage:
  python experiments/lock_snn_rr.py --validate '{"threshold":0.1,"n_timesteps":32,
    "orders":[0,1,2],"hidden":128,"scheme":"sqrt","lr":0.004,"epochs":100}' --seeds 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neurocardio.config import Config
from neurocardio.data.dataset import build_external_split_rr, build_split_rr
from neurocardio.data.splits import DS1_RECORDS, DS2_RECORDS
from neurocardio.encoding.beat import encode_beat
from neurocardio.models.snn import SNNClassifier
from neurocardio.train.loop import resolve_device, set_seed

N_CLASSES = 5
N_RR = 3
GOAL = {"VEB_sens": 0.90, "VEB_ppv": 0.50, "SVEB_sens": 0.45, "synops": 25000}
BUFFER = {"VEB_sens": 0.02, "VEB_ppv": 0.03, "SVEB_sens": 0.03}
VAL_RECORDS = ["201", "207", "223"]
TRAIN_RECORDS = [r for r in DS1_RECORDS if r not in VAL_RECORDS]
CACHE = Path("runs/lock_rr_cache")
CACHE.mkdir(parents=True, exist_ok=True)


def raw_group(cfg, records, tag):
    fb, fl, fr = (CACHE / f"{tag}_{s}.npy" for s in ("beats", "labels", "rr"))
    if fb.exists() and fl.exists() and fr.exists():
        return np.load(fb), np.load(fl), np.load(fr)
    beats, labels, rr = build_split_rr(cfg, records)
    np.save(fb, beats)
    np.save(fl, labels)
    np.save(fr, rr)
    return beats, labels, rr


def raw_external(cfg, name, record_dir, lead):
    """Cache raw (beats, labels, rr) for an external database, resampled to config fs."""
    fb, fl, fr = (CACHE / f"ext_{name}_{s}.npy" for s in ("beats", "labels", "rr"))
    if fb.exists() and fl.exists() and fr.exists():
        return np.load(fb), np.load(fl), np.load(fr)
    beats, labels, rr = build_external_split_rr(cfg, record_dir, lead_index=lead)
    np.save(fb, beats)
    np.save(fl, labels)
    np.save(fr, rr)
    return beats, labels, rr


def subsample_normal(beats, labels, rr, n_cap, seed=0):
    """Keep every non-Normal beat; cap the Normal class at n_cap. Used when adding an
    external database to TRAINING: the rare classes (SVEB, VEB) are the point, and
    MIT-BIH already supplies plenty of Normal, so we do not need all of the external
    Normal beats (they only slow training). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(N_CLASSES):
        idx = np.where(labels == c)[0]
        if c == 0 and len(idx) > n_cap:
            idx = rng.choice(idx, n_cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return beats[keep], labels[keep], rr[keep]


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
    return tp / max(tp + fn, 1), tp / max(tp + fp, 1)


def fit_bias(logits, labels):
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


def train_model(trx, trr, trl, hidden, in_features, weight, lr, epochs, batch, device, seed):
    set_seed(seed)
    model = SNNClassifier(
        in_features=in_features, hidden=hidden, n_classes=N_CLASSES, n_rr=N_RR
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=device))
    n = len(trx)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            lf(model(trx[idx], trr[idx]), trl[idx]).backward()
            opt.step()
    return model


def logits_of(model, x, rr, batch):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(x[i : i + batch], rr[i : i + batch]).cpu())
    return torch.cat(out).numpy()


def eval_split(model, x, rr, labels, bias, batch):
    """Apply the frozen bias to a split and return VEB/SVEB sensitivity and PPV."""
    pred = (logits_of(model, x, rr, batch) + bias).argmax(1)
    vs, vp = _metrics(pred, labels, 2)
    ss, sp = _metrics(pred, labels, 1)
    return {
        "VEB_sens": round(vs, 4),
        "VEB_ppv": round(vp, 4),
        "SVEB_sens": round(ss, 4),
        "SVEB_ppv": round(sp, 4),
        "n": int(len(labels)),
    }


def synops_per_beat(model, x_sample):
    """Hidden/output synaptic ops per beat, plus the one-shot RR->hidden projection."""
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
    ops = fc1_in * model.fc1.out_features + fc2_in * model.fc2.out_features
    rr_ops = b * N_RR * model.fc1.out_features  # dense RR->hidden, applied once per beat
    return (ops + rr_ops) / b


def run_once(p, data, device, seed):
    orders = p["orders"]
    hidden, batch = p.get("hidden", 128), p.get("batch", 512)
    lr, epochs, scheme = p.get("lr", 0.004), p.get("epochs", 100), p.get("scheme", "sqrt")
    trx, trr, trl_t, vax, var, val_l, tex, ter, tel, external = data
    in_features = 2 * len(orders)
    weight = class_weights(trl_t.cpu().numpy(), scheme)

    t0 = time.time()
    model = train_model(
        trx, trr, trl_t, hidden, in_features, weight, lr, epochs, batch, device, seed
    )
    val_logits = logits_of(model, vax, var, batch)
    bias, feasible_on_val = fit_bias(val_logits, val_l)
    ds2 = eval_split(model, tex, ter, tel, bias, batch)  # frozen operating point
    synops = synops_per_beat(model, tex[:2048])
    # Same frozen bias applied to each external database (no re-tuning).
    ext_res = {n: eval_split(model, ex, er, el, bias, batch) for n, (ex, er, el) in external.items()}
    res = {
        **{
            k: p.get(k)
            for k in ["threshold", "n_timesteps", "orders", "hidden", "scheme", "lr", "epochs"]
        },
        "seed": seed,
        "bias": [round(float(x), 3) for x in bias],
        "feasible_on_val": bool(feasible_on_val),
        "VEB_sens": ds2["VEB_sens"],
        "VEB_ppv": ds2["VEB_ppv"],
        "SVEB_sens": ds2["SVEB_sens"],
        "SVEB_ppv": ds2["SVEB_ppv"],
        "synops": round(synops, 1),
        "external": ext_res,
        "seconds": round(time.time() - t0, 1),
    }
    res["pass"] = (
        ds2["VEB_sens"] >= GOAL["VEB_sens"]
        and ds2["VEB_ppv"] >= GOAL["VEB_ppv"]
        and ds2["SVEB_sens"] >= GOAL["SVEB_sens"]
        and synops <= GOAL["synops"]
    )
    return res, model.state_dict()


def load_data(cfg, p, device, external_specs=None, augment_specs=None, n_cap=20000):
    trb, trl, trr = raw_group(cfg, TRAIN_RECORDS, "trainsub")
    train_tag = "trainsub"
    # Augmentation: add external databases to the TRAINING set (test stays MIT-BIH DS2).
    for name, record_dir, lead in augment_specs or []:
        ab, al, ar = raw_external(cfg, name, record_dir, lead)
        ab, al, ar = subsample_normal(ab, al, ar, n_cap)
        trb, trl, trr = (
            np.concatenate([trb, ab]),
            np.concatenate([trl, al]),
            np.concatenate([trr, ar]),
        )
        train_tag += f"_{name}"
    vab, val, var = raw_group(cfg, VAL_RECORDS, "val")
    teb, tel, ter = raw_group(cfg, DS2_RECORDS, "test")
    thr, T, orders = p["threshold"], p["n_timesteps"], p["orders"]
    # standardize RR features on (possibly augmented) train stats only
    mu, sd = trr.mean(0), trr.std(0) + 1e-6
    to_t = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32)).to(device)  # noqa: E731
    trx = torch.from_numpy(enc(trb, thr, T, orders, train_tag)).to(device)
    vax = torch.from_numpy(enc(vab, thr, T, orders, "val")).to(device)
    tex = torch.from_numpy(enc(teb, thr, T, orders, "test")).to(device)
    trl_t = torch.from_numpy(trl.astype(np.int64)).to(device)
    external = {}
    for name, record_dir, lead in external_specs or []:
        eb, el, er = raw_external(cfg, name, record_dir, lead)
        ex_x = torch.from_numpy(enc(eb, thr, T, orders, f"ext_{name}")).to(device)
        external[name] = (ex_x, to_t(er), el.astype(np.int64))
    return (trx, to_t(trr), trl_t, vax, to_t(var), val.astype(np.int64),
            tex, to_t(ter), tel.astype(np.int64), external)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/lock_rr_results.json")
    ap.add_argument(
        "--external",
        default="",
        help="comma list of name:dir:lead to TEST on, e.g. svdb:data/svdb:0,incart:data/incartdb:1",
    )
    ap.add_argument(
        "--augment",
        default="",
        help="comma list of name:dir:lead to ADD to TRAINING (do not also --external the same db)",
    )
    ap.add_argument("--augment-n-cap", type=int, default=20000, help="cap on Normal beats per aug db")
    args = ap.parse_args()
    device = resolve_device("auto")
    cfg = Config()

    def parse_specs(s):
        out = []
        for item in filter(None, s.split(",")):
            name, record_dir, lead = item.rsplit(":", 2)
            out.append((name, record_dir, int(lead)))
        return out

    external_specs = parse_specs(args.external)
    augment_specs = parse_specs(args.augment)
    ext_names = [n for n, _, _ in external_specs]
    print(
        f"device={device} train={len(TRAIN_RECORDS)} MIT-BIH recs "
        f"+ augment={[n for n, _, _ in augment_specs]}, val={VAL_RECORDS}, external={ext_names}",
        flush=True,
    )

    p = json.loads(args.validate)
    data = load_data(cfg, p, device, external_specs, augment_specs, args.augment_n_cap)
    rows = []
    for s in range(args.seeds):
        r, _ = run_once(p, data, device, s)
        rows.append(r)
        print(
            f"seed {s}: DS2 VEB {r['VEB_sens']}/{r['VEB_ppv']} SVEB {r['SVEB_sens']} "
            f"syn={r['synops']} pass={r['pass']} (valfeasible={r['feasible_on_val']}, {r['seconds']}s)",
            flush=True,
        )
        for n, m in r.get("external", {}).items():
            print(
                f"    [{n}] VEB {m['VEB_sens']}/{m['VEB_ppv']} SVEB {m['SVEB_sens']}/{m['SVEB_ppv']} "
                f"(n={m['n']})",
                flush=True,
            )
        Path(args.out).write_text(json.dumps(rows, indent=2))
    arr = {
        k: np.array([r[k] for r in rows])
        for k in ["VEB_sens", "VEB_ppv", "SVEB_sens", "synops"]
    }
    print(
        "DS2  ALL PASS:",
        all(r["pass"] for r in rows),
        "| mean:",
        {k: round(float(v.mean()), 4) for k, v in arr.items()},
        "| min:",
        {k: round(float(v.min()), 4) for k, v in arr.items()},
        flush=True,
    )
    for n in ext_names:
        exarr = {
            k: np.array([r["external"][n][k] for r in rows])
            for k in ["VEB_sens", "VEB_ppv", "SVEB_sens", "SVEB_ppv"]
        }
        print(
            f"{n:4s} mean:",
            {k: round(float(v.mean()), 4) for k, v in exarr.items()},
            "| min:",
            {k: round(float(v.min()), 4) for k, v in exarr.items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
