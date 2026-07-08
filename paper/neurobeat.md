# NeuroBeat: Energy-Accounted Spiking Networks for Honest Inter-Patient Ventricular Ectopic Beat Detection

**Author:** Valerio Merlini (independent researcher)
**Contact:** github.com/Talch87/neuro-beat
**Status:** Preprint draft, content-complete. All results are from locked 5-seed runs
(val-locked, DS2 frozen). Pending before posting: LaTeX build and a final author-list
check on the SNN-ECG references marked †.

---

## Abstract

Spiking neural networks (SNNs) are attractive for always-on cardiac monitoring
because their event-driven computation maps onto low-power neuromorphic
hardware. Yet much of the SNN-ECG literature reports accuracy under protocols
that quietly inflate it — intra-patient splits, or decision thresholds tuned on
the test set — and rarely accounts for the energy that motivates the approach in
the first place. We present **NeuroBeat**, a compact leaky-integrate-and-fire
(LIF) SNN for ventricular ectopic beat (VEB) detection, evaluated under a
deliberately conservative protocol: a patient-disjoint inter-patient split
(MIT-BIH DS1→DS2), an operating point fit **only** on a DS1 validation holdout
and then **frozen** for all test data, and an explicit per-beat synaptic-operation
(SynOps) energy budget. At a single operating point the model faces a genuine sensitivity–precision
tradeoff (VEB sensitivity 0.89 at PPV 0.49, *or* 0.86 at 0.68 — not both), which
we trace to calibration variance rather than model capacity. We break it with a
**gated-ensemble two-stage cascade**: a genuinely sparse high-recall screener
(5,987 SynOps/beat) runs on every beat and gates a 3-seed ensemble confirmer that
runs only on the ~27% flagged candidates. The cascade reaches **VEB sensitivity
0.923 at PPV 0.616 on DS2 within 23,385 SynOps/beat** — meeting sensitivity
≥0.90, PPV ≥0.60, and the ≤25k energy budget simultaneously — and holds VEB
sensitivity ≥0.90 across all three databases (DS2, SVDB, INCART) under one frozen
operating point. Along the way we show that count-pooled encoding makes energy
timestep-independent, so time resolution is a free discrimination lever but
short networks are not cheap. Finally, we report a negative result:
supraventricular (SVEB) detection is data- and lead-limited on a single lead
(DS2 SVEB sensitivity ~0.13, rising to 0.62 on 12-lead INCART),
which we address separately with an SVEB specialist rather than by compromising
the VEB operating point. Code, weights, and a live results dashboard are public.

---

## 1. Introduction

Continuous ECG monitoring on wearable and implantable devices is power-limited:
the detector must run for days on a coin cell. Ventricular ectopic beats (VEBs)
are clinically important — frequent VEBs and ventricular runs precede dangerous
arrhythmias — and a good always-on VEB detector must be both **sensitive** (miss
few true VEBs) and **cheap** (little energy per beat).

Spiking neural networks are a natural fit: they compute with sparse binary
events, and on neuromorphic substrates energy scales with the number of
synaptic operations actually performed, not with a fixed clock. This has driven
a wave of SNN-ECG work. However, two problems recur:

1. **Optimistic evaluation.** Reported accuracy is often inflated by
   intra-patient splits (the same patient's beats appear in train and test) or
   by selecting the decision threshold on the test set. Both leak information
   that a deployed device never has.
2. **Missing energy accounting.** The energy argument is the reason to use an
   SNN, yet many papers report only accuracy, or parameter counts, with no
   per-inference operation budget.

We take the opposite stance on both. Our contributions:

- **An honest, val-locked inter-patient protocol.** We use the de Chazal
  DS1/DS2 patient-disjoint split, fit the operating point (per-class logit
  biases) on a DS1 validation holdout, and freeze it for DS2 and all external
  data. The test set chooses nothing.
- **Explicit energy accounting.** We report per-beat SynOps and hold every
  configuration to a ≤25k-SynOps/beat budget. We identify **time resolution**
  (number of SNN timesteps) as a cheap discrimination lever.
- **Cross-database external validation.** The single frozen operating point is
  applied unchanged to two databases the model never trained on (MIT-BIH SVDB,
  INCART), quantifying real generalization.
- **A two-stage energy-gated cascade** that breaks the single-operating-point
  sensitivity–precision tradeoff at low average energy.
- **A characterized negative result** for supraventricular beats: SVEB is
  data-/lead-limited, not an architecture failure, and is better pursued by a
  dedicated specialist than by degrading the VEB operating point.

Everything is reproducible: code, frozen weights, per-seed logs, and a public
dashboard.

---

## 2. Related Work

**Inter-patient benchmarks and evaluation hygiene.** The AAMI EC57 standard
[AAMI-EC57] defines the five beat classes and the sensitivity/PPV reporting
convention. de Chazal et al. [deChazal2004] introduced the patient-disjoint
DS1/DS2 split (44 MIT-BIH records, ~50k beats each) that we adopt. A 2025
systematic review [Silva2025] surveys 2017–2024 arrhythmia classifiers and finds
that, despite high reported accuracy, few studies implement patient-independent
partitioning or account for embedded constraints — the exact gap our protocol
targets. NeuroBeat contributes an explicitly val-locked operating point (fit on a
DS1 holdout, never on DS2) and cross-database external validation on top of the
DS1/DS2 convention.

**SNNs for ECG / arrhythmia.** SNNs are attractive for ECG because event-driven
computation maps onto low-power neuromorphic hardware. Prior work reports strong
MIT-BIH accuracy with delta-modulation encoding of the ECG and its derivatives
[SNN-ECG-BSPC2021], spike-driven neuromorphic processors for wearable ECG
[Chu2022], hardware/software co-design [SparrowSNN2024], and axonal-delay SNNs
[AxonalDelays2025]. Our encoding (count-pooled delta channels over signal orders)
is in this family, but most of these report intra-patient or test-tuned accuracy;
we instead report val-locked inter-patient numbers with explicit energy budgets
and a negative result for the supraventricular class.

**SNN training and energy.** We train with surrogate-gradient
backpropagation-through-time [Neftci2019] via snnTorch [Eshraghian2023]. On
neuromorphic substrates, per-inference energy scales with synaptic operations;
reported ECG figures show ~4.68 µJ/inference on a neuromorphic chip versus
~450 µJ on a conventional processor [Chu2022], and general-purpose neuromorphic
cores such as Loihi [Davies2018] motivate the SynOps proxy we budget against.

**Cascaded / gated inference.** Two-stage screener–confirmer and energy-gated
designs are standard elsewhere for trading average cost against accuracy; we
apply the pattern to keep an ensemble-grade confirmer affordable by running it
only on flagged candidates.

**Data.** MIT-BIH [Moody2001], its Supraventricular database (SVDB), and the
St. Petersburg INCART 12-lead database are distributed via PhysioNet
[Goldberger2000].

---

## 3. Data

**MIT-BIH Arrhythmia Database (primary).** 48 two-lead records, 360 Hz. Beats
are mapped to the five AAMI classes: N (normal), S (supraventricular ectopic,
"SVEB"), V (ventricular ectopic, "VEB"), F (fusion), Q (unknown). We use the
de Chazal inter-patient split: **DS1** for training, **DS2** for test, with no
patient in both. From DS1 we hold out records **201, 207, 223** as a
patient-disjoint **validation** set for operating-point selection.

**External databases (test-only generalization).** MIT-BIH Supraventricular
Arrhythmia Database (**SVDB**, 128 Hz) and St. Petersburg INCART 12-lead
database (**INCART**, 257 Hz). Signals are resampled to 360 Hz
(polyphase, `scipy.signal.resample_poly` on the reduced gcd ratio) and
annotation sample positions rescaled accordingly. Lead selection: MIT-BIH and
SVDB lead 0; INCART lead II (index 1).

**Beat segmentation and RR features.** Beats are windowed around each annotated
R-peak. For each beat we also compute three patient-normalized RR-interval
features (previous, current, and ratio), standardized using training-set
statistics only.

---

## 4. Methods

### 4.1 Spike encoding

Each beat window is converted to a spike tensor by a **count-pooled delta
encoding**. For each signal order in `orders` (order 0 = the signal, order 1 =
its first derivative), threshold crossings of magnitude `θ = 0.12` are pooled
into `T` time bins, producing two channels (up/down) per order. With
`orders = [0, 1]` this yields `2·|orders| = 4` input channels over `T`
timesteps. Time resolution `T` is a key lever: raising it sharpens
discrimination at almost no extra energy (Section 6).

### 4.2 Network

A compact two-layer LIF network (`SNNClassifier`): `fc1` projects the 4 input
channels to `H = 128` hidden LIF units; `fc2` projects hidden spikes to the 5
class logits. The three RR features are projected once through a dense map into
the hidden state. Readout is by accumulated membrane potential. The network is
trained by surrogate-gradient backpropagation-through-time (snnTorch) with
Adam (lr 4e-3, 100 epochs) and `sqrt`-scaled class weights to counter class
imbalance.

### 4.3 Val-locked operating point

The network outputs 5 logits; the decision applies a fixed per-class **bias**
vector `b = [0, b_S, b_V, −12, −12]` (F and Q suppressed) and takes the argmax.
`(b_S, b_V)` is chosen on the **DS1 validation holdout only** and then **frozen**
for DS2 and every external database. We consider three val-fit strategies:

- **sens-first** — maximize VEB PPV subject to VEB sensitivity ≥ 0.90;
- **ppv-first** — maximize VEB sensitivity subject to VEB PPV ≥ 0.60;
- **balanced** — maximize VEB F1.

The frozen NeuroBeat-VEB v1 uses **sens-first** (missing ventricular beats is
the costlier error), and we report the full sensitivity–PPV frontier so the
tradeoff is explicit.

### 4.4 Energy proxy

Per-beat energy is the synaptic-operation (SynOps) count:
`SynOps = (Σ_t input_spikes)·H + (Σ_t hidden_spikes)·C + n_RR·H`,
where `C = 5` classes and the last term is the one-shot RR→hidden projection.
The first term dominates and is encoding-dependent, which is why sparse,
low-order input and modest `T` matter. Budget: **≤ 25,000 SynOps/beat**.

### 4.5 Two-stage cascade

To obtain high sensitivity **and** high precision without paying full energy on
every beat, we cascade two independent networks:

- **Stage 1 (screener)** — a cheaper SNN at half the timesteps (`T = 32`),
  operating point fit for high VEB **recall** (~0.97 on val). Runs on **every**
  beat; emits candidate VEBs.
- **Stage 2 (confirmer)** — the frozen v1 (`T = 64`) at a high-**precision**
  operating point, run **only** on Stage-1 candidates. A beat is declared VEB
  iff both stages fire.

Stage 1 must be a *different* network from v1 (different `T` ⇒ different
encoding ⇒ partly independent errors); otherwise re-judging its own candidates
adds nothing. Average energy is
`SynOps(stage1) + flag_rate · SynOps(stage2 on candidates)`, and because the
candidate set is small the heavier confirmer costs little on average. Both
operating points are fit on val only.

---

## 5. Experiments

All results use 5 training seeds. The frozen seed is selected by **validation**
performance; test numbers are reported for that frozen model and, for stability,
as the 5-seed DS2 mean/min. No number below was used to choose an operating
point on the split it is reported on.

### 5.1 Single-stage VEB detection (val-locked)

DS2 VEB detection under each val-fit strategy, 5 seeds, reported as mean (min):

| Operating point | VEB sensitivity | VEB PPV | SynOps/beat |
|---|---|---|---|
| **sens-first** | 0.905 (0.878) | 0.539 (0.490) | ~14.3k |
| **balanced**   | 0.857 (0.790) | 0.679 (0.517) | ~14.3k |
| ppv-first      | 0.845 (0.771) | 0.700 (0.583) | ~14.3k |

The single operating point buys sensitivity **or** precision, not both: at
sensitivity ≈0.90 the PPV is ~0.54, and reaching PPV ~0.68 costs ~4 points of
sensitivity. All configurations sit comfortably under the 25k-SynOps budget.

**Sensitivity–PPV frontier (DS2, oracle sweep).** PPV achievable at each
sensitivity level (range over the 5 seeds; 2 of 5 seeds top out just below 0.90
sensitivity on DS2, an artifact of the small validation holdout — Section 6):

| sensitivity ≥ | 0.80 | 0.85 | 0.90 | 0.92 |
|---|---|---|---|---|
| VEB PPV | 0.57–0.78 | 0.57–0.69 | 0.56–0.65 | 0.48–0.57 |

Even the oracle frontier does not offer a single point at both ≥0.90 sensitivity
and ≥0.60 PPV reliably across seeds — motivating the cascade (Section 5.3).

### 5.2 Cross-database generalization

The frozen NeuroBeat-VEB v1 (sens-first operating point, fit on DS1-val) applied
**unchanged** to three databases:

| Database | Beats | VEB sensitivity | VEB PPV |
|---|---|---|---|
| MIT-BIH DS2 | ~50k | 0.894 | 0.490 |
| MIT-BIH SVDB | ~184k | 0.892 | 0.361 |
| INCART (12-lead) | ~176k | 0.880 | 0.736 |

VEB **sensitivity holds at 0.88–0.89 across all three databases** under one
frozen operating point — the detector transfers. PPV varies with each database's
class mix (lower on the supraventricular-dense SVDB, higher on INCART).

### 5.3 Two-stage cascade

A naive cascade — a T32 screener gating the single frozen v1 — reaches DS2
0.883 / 0.622 but at **19.4k** SynOps, *more* than single-stage v1, because
count-pooling makes the T32 screener as expensive as the T64 confirmer
(Section 6). We therefore use a **sparse** screener (2× fewer hidden units,
higher spike threshold; 5,987 SynOps/beat) and, since a K-seed ensemble is what
actually reaches the accuracy target, a **gated 3-seed ensemble confirmer** run
only on the ~27% of beats the screener flags. The ensemble confirmer requires no
extra training — the three seed models are already trained for the single-stage
study — and its members' logits are combined by averaging.

Both operating points are fit on DS1-val only (screener recall target 0.97;
confirmer maximizes cascade PPV subject to cascade sensitivity ≥ 0.90). The
frozen result:

| Database | VEB sensitivity | VEB PPV | flag rate |
|---|---|---|---|
| **MIT-BIH DS2** | **0.923** | **0.616** | 0.271 |
| MIT-BIH SVDB | 0.904 | 0.377 | 0.345 |
| INCART | 0.901 | 0.835 | 0.241 |

Average energy: `5,987 + 0.271 · 3 · 21,433 = 23,385` SynOps/beat.
The cascade **meets all three targets simultaneously on DS2** — sensitivity
≥0.90, PPV ≥0.60, and ≤25k SynOps — while holding VEB sensitivity ≥0.90 across
all three databases. Ensemble composition is robust: K∈{2,3,5} and different
3-seed subsets all give DS2 sensitivity ≥0.90 at PPV 0.59–0.64.

**Accuracy–energy Pareto (DS2 VEB).**

| Model | sens / PPV | SynOps/beat |
|---|---|---|
| Single-stage v1 (sens-first) | 0.894 / 0.490 | 14.2k |
| Naive T32 cascade | 0.883 / 0.622 | 19.4k |
| 5-seed ensemble (every beat) | 0.932 / 0.595 | ~71k |
| **Gated-ensemble cascade** | **0.923 / 0.616** | **23.4k** |

### 5.4 Baselines on the same split

We train non-spiking CNN and LSTM baselines under the **identical** protocol —
same DS1-train, same beat+RR inputs, same DS1-val sens-first operating point,
same frozen DS2 test, 5 seeds:

| Model | DS2 VEB sens | DS2 VEB PPV | Ops/beat | Params |
|---|---|---|---|---|
| CNN (1-D, beat+RR) | 0.945 | 0.434 | 356k MACs | 2.9k |
| LSTM (beat+RR) | 0.940 | 0.178 | 4.26M MACs | 17.5k |
| SNN single-stage v1 | 0.894 | 0.490 | 14.2k SynOps | 18k |
| **SNN gated-ensemble** | **0.923** | **0.616** | **23.4k SynOps** | — |

Two observations. First, the dense baselines exhibit the *same*
sensitivity–precision tradeoff — under the sens-first operating point they reach
high sensitivity (~0.94) but low PPV (0.43 CNN, 0.18 LSTM) — confirming the
tradeoff is a property of single-lead VEB detection, not of spiking networks.
Second, on the operation count that stands in for energy, the SNN is far cheaper:
the CNN and LSTM require ~356k and ~4.26M multiply-accumulates per beat, whereas
the SNN performs ~14–23k synaptic operations. A MAC and a SynOp are not identical
units — a SynOp is a single accumulate triggered by a spike, and on neuromorphic
hardware is cheaper than a clocked MAC — so the ~15–180× operation-count gap is a
conservative lower bound on the energy advantage. The gated-ensemble SNN also
attains the best precision (0.616) of any model at comparable sensitivity.

### 5.5 Supraventricular beats: a single-lead limitation

SVEB detection is hard on one lead: supraventricular ectopic beats often differ
from normal beats mainly in timing, not morphology, and DS1 contains few of them.
As an incidental output of the VEB model, DS2 SVEB sensitivity is ~0.13; adding
supraventricular-rich svdb to training raises it to ~0.24, and the T96 lever to
~0.27 — still far from usable.

We then trained a **dedicated SVEB specialist** (T96, svdb + INCART augmentation,
SVEB-only operating point, 5 seeds). It does **not** yield a usable detector:

| Seed regime | DS2 SVEB sens | DS2 SVEB PPV | DS2 VEB sens |
|---|---|---|---|
| calibrated (seeds 0–2) | 0.16–0.27 | 0.07–0.15 | collapses |
| degenerate (seeds 3–4) | 0.82–0.94 | 0.04–0.06 | 0.18–0.24 |

The high-sensitivity seeds reach it only by flagging almost everything as SVEB
(PPV ≤0.06) and in doing so cannibalize the ventricular class. DS2 SVEB PPV never
exceeds ~0.15 at any operating point, and the result is wildly seed-unstable. The
same architecture reached SVEB sensitivity 0.62 on 12-lead INCART, so the limit
is **lead count and data, not capacity**. Multi-lead input (MIT-BIH provides two
leads; supraventricular beats separate better on some leads) is the natural next
lever and is left to future work. This is a genuine negative result: on a single
lead, supraventricular detection at usable precision is out of reach here, which
is precisely why it must not be allowed to compromise the VEB operating point.

---

## 6. Discussion

**Count-pooling makes energy timestep-independent — with two consequences.**
Under count-pooled delta encoding the total number of input spikes is
approximately conserved across timesteps `T`; the dominant SynOps term
(input-spikes × hidden) therefore barely changes with `T`. (i) *Time resolution
is a free discrimination lever*: raising `T` from 32 to 64 sharpens separation of
ectopic morphologies at almost no energy cost. (ii) *You cannot build a cheap
screener by lowering `T`* — a T32 network costs essentially the same as a T64
one. A genuinely sparse screener must instead reduce input channels, hidden
units, or raise the spike threshold. This is why our two-stage screener is
sparse-by-construction rather than merely shorter.

**The shortfall of a single model is calibration variance, not capacity.** The
DS2 oracle frontier shows ≥0.90 sensitivity at ≥0.60 PPV is achievable, yet a
single model calibrated on the small validation holdout rarely lands there and
varies widely across seeds. A K-seed deep ensemble both averages away model
variance and calibrates cleanly, reaching the target corner
(0.918 / 0.634 on DS2 for a 3-seed ensemble, with ~0.90 sensitivity across all
three databases). The two-stage **gated-ensemble** cascade delivers this accuracy
within the energy budget by running the ensemble only on flagged candidates.

**The honest protocol costs headline accuracy but buys trust.** Freezing the
operating point on val rather than test lowers reported numbers relative to
test-tuned protocols, but the cross-database results show the frozen point
transfers, which test-tuned numbers cannot promise.

**Limitations.** Single-lead SVEB detection is inherently hard; our energy proxy
is SynOps, not measured hardware joules; and the small validation holdout makes
single-model operating points noisy (mitigated by the ensemble). A larger or
cross-validated DS1 holdout is likely to tighten single-model calibration.

---

## 7. Conclusion

NeuroBeat is a compact, energy-accounted spiking VEB detector evaluated under a
conservative, leak-free inter-patient protocol with cross-database external
validation. A two-stage cascade delivers high sensitivity and precision within a
25k-SynOps/beat budget. We release code, frozen weights, and a live dashboard,
and we treat supraventricular detection as a separate, data-limited problem.

---

## 8. Reproducibility

- Repository: github.com/Talch87/neuro-beat (AGPL-3.0 / commercial dual license)
- Live dashboard: talch87.github.io/neuro-beat
- Frozen artifact: `models/neurobeat-veb-v1/{weights.pt, operating_point.json}`
- Protocol harness: `experiments/freeze_veb_v1.py`, `experiments/gated_ensemble_veb.py`,
  `experiments/baselines_veb.py`
- All experiments: 5 seeds, operating point fit on DS1-val only.

---

## References

*Compiled and cross-checked via literature search (July 2026). Author lists for the
SNN-ECG works marked † should be confirmed against the source before submission.*

- **[AAMI-EC57]** ANSI/AAMI EC57. *Testing and Reporting Performance Results of
  Cardiac Rhythm and ST-Segment Measurement Algorithms.* Association for the
  Advancement of Medical Instrumentation.
- **[deChazal2004]** P. de Chazal, M. O'Dwyer, R. B. Reilly. "Automatic
  classification of heartbeats using ECG morphology and heartbeat interval
  features." *IEEE Trans. Biomed. Eng.*, 51(7):1196–1206, 2004.
  doi:10.1109/TBME.2004.827359.
- **[Silva2025]** G. Silva, P. Silva, G. Moreira, V. Freitas, J. Gertrudes,
  E. Luz. "A Systematic Review of ECG Arrhythmia Classification: Adherence to
  Standards, Fair Evaluation, and Embedded Feasibility." arXiv:2503.07276, 2025.
- **[Moody2001]** G. B. Moody, R. G. Mark. "The impact of the MIT-BIH Arrhythmia
  Database." *IEEE Eng. Med. Biol. Mag.*, 20(3):45–50, 2001.
- **[Goldberger2000]** A. L. Goldberger et al. "PhysioBank, PhysioToolkit, and
  PhysioNet." *Circulation*, 101(23):e215–e220, 2000.
- **[Neftci2019]** E. O. Neftci, H. Mostafa, F. Zenke. "Surrogate Gradient
  Learning in Spiking Neural Networks." *IEEE Signal Process. Mag.*,
  36(6):51–63, 2019.
- **[Eshraghian2023]** J. K. Eshraghian et al. "Training Spiking Neural Networks
  Using Lessons From Deep Learning." *Proc. IEEE*, 111(9):1016–1054, 2023.
- **[Davies2018]** M. Davies et al. "Loihi: A Neuromorphic Manycore Processor
  with On-Chip Learning." *IEEE Micro*, 38(1):82–99, 2018.
- **[SNN-ECG-BSPC2021]** † "Energy efficient ECG classification with spiking
  neural network." *Biomed. Signal Process. Control*, 2021.
  (ScienceDirect S1746809420303098.)
- **[Chu2022]** † "A Neuromorphic Processing System With Spike-Driven SNN
  Processor for Wearable ECG Classification." *IEEE Trans. Biomed. Circuits
  Syst.*, 2022. (PubMed 35802543.)
- **[SparrowSNN2024]** † "SparrowSNN: A Hardware/Software Co-design for Energy
  Efficient ECG Classification." arXiv:2406.06543, 2024.
- **[AxonalDelays2025]** † "Robust ECG signal classification using spiking neural
  networks with axonal delays." *Neurocomputing*, 2025.
  (ScienceDirect S0925231225029315.)
