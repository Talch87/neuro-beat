# NeuroBeat-VEB v1 — model card

Compact spiking (LIF) detector for **ventricular ectopic beats (VEB)** on ECG,
evaluated under an honest inter-patient protocol (MIT-BIH DS1→DS2), with the
operating point fit on a DS1 validation holdout and **frozen** for all test data.

## Artifacts

| File | What it is |
|---|---|
| `weights.pt` | Single-stage v1 (seed 2): T64, orders[0,1], hidden 128, thr 0.12 |
| `operating_point.json` | v1 sens-first operating point + frozen DS2/SVDB/INCART metrics |
| `screener_weights.pt` | Sparse screener: T32, orders[0,1], hidden 64, thr 0.18 |
| `confirmer_seed{0,1,2}.pt` | 3-seed ensemble confirmer (T64 v1 architecture) |
| `gated_ensemble.json` | Gated-ensemble cascade config + frozen metrics (recommended) |
| `cascade.json`, `stage1_weights.pt` | Naive T32 cascade (superseded; kept for the paper's Pareto) |

## Recommended: gated-ensemble cascade

Sparse screener on every beat gates the 3-seed ensemble confirmer on the ~27%
flagged candidates. **Meets all three targets on DS2 simultaneously:**

| Database | VEB sensitivity | VEB PPV |
|---|---|---|
| MIT-BIH DS2 | 0.923 | 0.616 |
| MIT-BIH SVDB | 0.904 | 0.377 |
| INCART | 0.901 | 0.835 |

- **Average energy:** 23,385 SynOps/beat (screener 5,987 + 0.271 · 3 · 21,433) ≤ 25k budget.
- Both operating points fit on DS1-val only; DS2/SVDB/INCART frozen.

## Accuracy–energy Pareto (DS2 VEB)

| Model | sens / PPV | SynOps/beat |
|---|---|---|
| Single-stage v1 (sens-first) | 0.894 / 0.490 | 14.2k |
| Naive T32 cascade | 0.883 / 0.622 | 19.4k |
| 5-seed ensemble (every beat) | 0.932 / 0.595 | ~71k |
| **Gated-ensemble cascade** | **0.923 / 0.616** | **23.4k** |

## Reproduce

```
python experiments/freeze_veb_v1.py --seeds 5      # single-stage + cache seeds
python experiments/gated_ensemble_veb.py           # gated-ensemble cascade
```

Not for clinical use. Trained/evaluated on public research databases only.
