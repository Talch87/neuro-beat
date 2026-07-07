from math import gcd

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


def resample_signal(signal, native_fs: int, target_fs: int) -> np.ndarray:
    """Resample a 1-D signal from native_fs to target_fs (polyphase).

    Used to bring other-database records (svdb 128 Hz, INCART 257 Hz) onto the
    MIT-BIH 360 Hz grid so the same sample-indexed beat windows apply. Annotation
    sample positions must be rescaled separately (see records.load_record)."""
    x = np.asarray(signal, dtype=np.float64)
    if native_fs == target_fs:
        return x
    g = gcd(int(native_fs), int(target_fs))
    return resample_poly(x, int(target_fs) // g, int(native_fs) // g)


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
