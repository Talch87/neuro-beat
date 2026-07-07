import numpy as np

AAMI_CLASSES = ["N", "SVEB", "VEB", "F", "Q"]
_CLASS_INDEX = {c: i for i, c in enumerate(AAMI_CLASSES)}

# MIT-BIH beat symbol -> AAMI class (per EC57). Non-beat markers map to None.
_SYMBOL_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "SVEB",
    "a": "SVEB",
    "J": "SVEB",
    "S": "SVEB",
    "V": "VEB",
    "E": "VEB",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


def symbol_to_aami(symbol: str):
    return _SYMBOL_MAP.get(symbol)


def segment_beats(
    signal, ann_samples, ann_symbols, window_before: int = 128, window_after: int = 128
):
    n = len(signal)
    beats, labels = [], []
    for s, sym in zip(ann_samples, ann_symbols):
        cls = symbol_to_aami(sym)
        if cls is None:
            continue
        start, end = s - window_before, s + window_after
        if start < 0 or end > n:
            continue
        beats.append(signal[start:end])
        labels.append(_CLASS_INDEX[cls])
    if not beats:
        return (
            np.zeros((0, window_before + window_after), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    return np.asarray(beats, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def beat_rr_features(
    ann_samples, ann_symbols, n_samples, window_before: int = 128, window_after: int = 128
):
    """Per-beat RR-interval features aligned 1:1 with segment_beats output.

    Returns [n_beats, 3]: [pre_RR / median_RR, post_RR / median_RR, pre_RR / post_RR].
    Ratios are patient-normalized (divided by the record's median RR), so they are
    dimensionless and inter-patient robust. Premature beats (SVEB, many VEB) have a
    short pre_RR, so pre_RR/median_RR < 1 -- the timing cue morphology alone lacks.
    """
    s = np.asarray(ann_samples, dtype=np.float64)
    is_beat = np.array([symbol_to_aami(sym) is not None for sym in ann_symbols], dtype=bool)
    beat_samples = s[is_beat]
    diffs = np.diff(beat_samples)
    median_rr = float(np.median(diffs)) if len(diffs) else 1.0
    if median_rr <= 0:
        median_rr = 1.0
    pre = np.full(len(beat_samples), median_rr)
    post = np.full(len(beat_samples), median_rr)
    if len(diffs):
        pre[1:] = diffs
        post[:-1] = diffs

    feats = []
    bi = 0
    for samp, sym in zip(ann_samples, ann_symbols):
        if symbol_to_aami(sym) is None:
            continue
        p, q = pre[bi], post[bi]
        bi += 1
        start, end = samp - window_before, samp + window_after
        if start < 0 or end > n_samples:
            continue
        feats.append([p / median_rr, q / median_rr, p / max(q, 1.0)])
    if not feats:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(feats, dtype=np.float64)
