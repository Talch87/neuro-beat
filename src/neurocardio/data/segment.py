import numpy as np

AAMI_CLASSES = ["N", "SVEB", "VEB", "F", "Q"]
_CLASS_INDEX = {c: i for i, c in enumerate(AAMI_CLASSES)}

# MIT-BIH beat symbol -> AAMI class (per EC57). Non-beat markers map to None.
_SYMBOL_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "SVEB", "a": "SVEB", "J": "SVEB", "S": "SVEB",
    "V": "VEB", "E": "VEB",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}


def symbol_to_aami(symbol: str):
    return _SYMBOL_MAP.get(symbol)


def segment_beats(signal, ann_samples, ann_symbols, window_before: int = 128,
                  window_after: int = 128):
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
        return (np.zeros((0, window_before + window_after), dtype=np.float64),
                np.zeros((0,), dtype=np.int64))
    return np.asarray(beats, dtype=np.float64), np.asarray(labels, dtype=np.int64)
