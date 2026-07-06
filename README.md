# neurocardio

SNN arrhythmia detector on single-lead ECG. Phase 1 proof-of-concept.

A tiny spiking neural network (LIF neurons, surrogate-gradient BPTT) classifies
ECG beats into the five AAMI EC57 classes (N / SVEB / VEB / F / Q), trained and
tested under a strict **inter-patient** split so the reported numbers survive
regulatory and investor scrutiny. A hardware-faithful level-crossing spike
encoder mirrors the analog front-end the eventual patch would use, and a
synaptic-operations (SynOps) proxy makes the "low-power neuromorphic" claim a
measurable number rather than an adjective.

## Setup
    uv venv --python 3.11
    uv pip install -e ".[dev]"
    uv run pytest -q

## Methodology (why the numbers are honest)

- **Inter-patient split (de Chazal DS1 → DS2).** Beats from one patient never
  appear in both train and test. Intra-patient (random) splits routinely report
  ~99% accuracy that collapses inter-patient; we report inter-patient from day
  one. The four paced records (102, 104, 107, 217) are excluded per AAMI EC57.
- **Per-class metrics, not accuracy.** Under ~89% normal-beat prevalence,
  overall accuracy is close to meaningless. We report per-class **sensitivity**
  (recall) and **positive predictivity** (PPV) for VEB and SVEB — the numbers a
  cardiologist and a notified body actually read.
- **Class weighting.** All three models are trained with inverse-frequency class
  weighting so the minority arrhythmia classes are learnable under the heavy
  N-class imbalance; without it every model collapses toward predicting "normal."

## Reproduce

    uv run neurocardio download --dest data/mitdb        # ~100 MB from PhysioNet
    uv run neurocardio train --config configs/snn.yaml  --out runs/snn.pt
    uv run neurocardio train --config configs/cnn.yaml  --out runs/cnn.pt
    uv run neurocardio train --config configs/lstm.yaml --out runs/lstm.pt

`configs/{snn,cnn,lstm}.yaml` fix `train.seed: 1337` for reproducibility. The SNN
uses the delta spike encoder; the CNN/LSTM baselines consume raw beats.

## Phase 1 results (MIT-BIH, inter-patient DS1→DS2)

DS1 train = 51,000 beats, DS2 test = 49,693 beats. All models trained 20 epochs
with inverse-frequency class weighting (seed 1337). Metrics per AAMI EC57.

| Model | Params | VEB Sens | VEB PPV | SVEB Sens | SVEB PPV | Overall Acc | SynOps/beat |
|-------|--------|----------|---------|-----------|----------|-------------|-------------|
| SNN (delta) | 1,029  | **0.808** | 0.320 | 0.407 | 0.064 | 0.629 | 33,277 |
| CNN1D       | 2,885  | 0.835 | 0.770 | 0.569 | 0.093 | 0.684 | n/a |
| LSTM        | 17,477 | 0.626 | 0.160 | 0.020 | 0.015 | 0.567 | n/a |

Split: de Chazal DS1 train / DS2 test. Paced records excluded. Metrics per AAMI EC57.

### Reading the table

- **The SNN is the Phase-1 de-risking result.** With only **1,029 parameters** —
  roughly a third of the CNN and 1/17th of the LSTM — it reaches **0.808 VEB
  sensitivity inter-patient**, essentially matching the CNN (0.835) and far ahead
  of the LSTM (0.626). Detecting ventricular ectopic beats is the clinically
  load-bearing case, and a network this small doing it on unseen patients is the
  evidence that a sub-milliwatt spiking chip is a credible target. The membrane-
  potential readout holds — the SNN learns rather than collapsing to "all normal."
- **PPV and SVEB are the honest weak spots.** Class weighting buys minority
  *sensitivity* at the cost of *positive predictivity*: the SNN over-calls VEB
  (PPV 0.320) and everyone struggles with SVEB (supraventricular beats are subtle
  on a single lead). The LSTM barely learns SVEB at all (sens 0.020) — a real
  finding, not a bug: a last-hidden-state LSTM over 256 raw samples is a weak
  inter-patient model here.
- **Overall accuracy is deliberately unimpressive** (0.57–0.68). Under ~89% N
  prevalence a "predict normal" model scores ~0.89; class weighting trades that
  headline number for the per-class sensitivity that actually matters. This is the
  point of reporting per-class metrics rather than accuracy.

**Phase-2 gate (VEB Sens ≥ 0.90 inter-patient) is not yet cleared** — expected for
a 20-epoch, untuned, 1k-parameter PoC on CPU. The visible next levers are: more
epochs / LR schedule, a recurrent or convolutional SNN topology, per-class weight
tuning to lift VEB PPV, and the delta-threshold sweep. The plan's job was to make
these numbers real, honest, and improvable — done.

_Reproduced on CPU (no CUDA), seed 1337. Per-model run: `configs/{snn,cnn,lstm}.yaml`._
