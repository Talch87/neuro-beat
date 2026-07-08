"""Non-spiking baselines (CNN, LSTM) for VEB, on the identical honest protocol.

Same DS1-train (pure), same DS1-val operating-point fit, same frozen DS2 test,
same beat+RR inputs, same VEB-only sens-first operating point as the SNN. This
is the apples-to-apples comparison for the paper's Section 5.4: does a spiking
network give up accuracy relative to conventional models, and at what energy?

Energy note: the SNN reports event-driven SynOps/beat; dense nets have no spikes,
so we report their multiply-accumulates (MACs)/beat and parameter count instead.
The units differ, but the operation counts are directly informative.

Usage:
  python experiments/baselines_veb.py --seeds 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from lock_snn_rr import (  # noqa: E402
    TRAIN_RECORDS,
    VAL_RECORDS,
    class_weights,
    raw_group,
)
from freeze_veb_v1 import op_sens_first, metr  # noqa: E402
from neurocardio.config import Config  # noqa: E402
from neurocardio.data.splits import DS2_RECORDS  # noqa: E402
from neurocardio.models.baselines import CNN1D, LSTMClassifier  # noqa: E402
from neurocardio.train.loop import resolve_device, set_seed  # noqa: E402

N_CLASSES, N_RR = 5, 3
BATCH, LR, EPOCHS = 512, 1e-3, 50


def macs_cnn(L):
    # conv1 (1->16, k7) over L; conv2 (16->32, k5) over L/2; head (32+3)->5
    return 16 * 7 * 1 * L + 32 * 5 * 16 * (L // 2) + (32 + N_RR) * N_CLASSES


def macs_lstm(L, hidden=64):
    # 4 gates x hidden x (in+hidden) per step x L steps; head (hidden+3)->5
    return 4 * hidden * (1 + hidden) * L + (hidden + N_RR) * N_CLASSES


def params(model):
    return sum(p.numel() for p in model.parameters())


def logits_of(model, x, rr, batch):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(x[i:i + batch], rr[i:i + batch]).cpu())
    return torch.cat(out).numpy()


def train(model, trx, trr, trl, weight, device, seed):
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lf = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=device))
    n = len(trx)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            lf(model(trx[idx], trr[idx]), trl[idx]).backward()
            opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/baselines_veb.json")
    args = ap.parse_args()
    device = resolve_device("auto")
    cfg = Config()

    trb, trl, trr = raw_group(cfg, TRAIN_RECORDS, "trainsub")
    vab, val_l, var = raw_group(cfg, VAL_RECORDS, "val")
    teb, tel, ter = raw_group(cfg, DS2_RECORDS, "test")
    L = trb.shape[1]
    mu, sd = trr.mean(0), trr.std(0) + 1e-6
    t = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)  # noqa: E731
    trx, vax, tex = t(trb), t(vab), t(teb)
    trrt, vart, tert = t((trr - mu) / sd), t((var - mu) / sd), t((ter - mu) / sd)
    trlt = torch.from_numpy(trl.astype(np.int64)).to(device)
    weight = class_weights(trl, "sqrt")
    print(f"device={device} beat_len={L} train={len(trb)} val={len(vab)} DS2={len(teb)}", flush=True)

    builders = {
        "CNN": (lambda: CNN1D(n_classes=N_CLASSES, n_rr=N_RR), macs_cnn(L)),
        "LSTM": (lambda: LSTMClassifier(n_classes=N_CLASSES, hidden=64, n_rr=N_RR), macs_lstm(L)),
    }
    summary = {}
    for name, (build, macs) in builders.items():
        rows = []
        for s in range(args.seeds):
            t0 = time.time()
            model = train(build(), trx, trrt, trlt, weight, device, s)
            (bias, _vs, _vp), feas = op_sens_first(logits_of(model, vax, vart, BATCH), val_l)
            ds2 = metr(logits_of(model, tex, tert, BATCH), tel, bias)
            rows.append({"seed": s, "val_feasible": bool(feas), **ds2})
            print(f"{name} seed {s}: DS2 VEB {ds2['VEB_sens']}/{ds2['VEB_ppv']} "
                  f"feas={feas} ({time.time()-t0:.0f}s)", flush=True)
        vs = np.array([r["VEB_sens"] for r in rows]); vp = np.array([r["VEB_ppv"] for r in rows])
        summary[name] = {
            "params": params(build()), "macs_per_beat": int(macs),
            "DS2_VEB_sens": {"mean": round(float(vs.mean()), 4), "min": round(float(vs.min()), 4)},
            "DS2_VEB_ppv": {"mean": round(float(vp.mean()), 4), "min": round(float(vp.min()), 4)},
            "seeds": rows,
        }
        print(f"{name}: DS2 VEB sens {vs.mean():.3f} (min {vs.min():.3f}) "
              f"ppv {vp.mean():.3f} | {params(build()):,} params | {macs:,} MACs/beat", flush=True)

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
