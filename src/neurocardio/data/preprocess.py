import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal, fs: int, low: float = 0.5, high: float = 40.0, order: int = 4
) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, np.asarray(signal, dtype=np.float64))


def normalize(signal) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    std = x.std()
    if std < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / std
