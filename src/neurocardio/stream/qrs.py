import numpy as np
from scipy.signal import find_peaks


def find_r_peaks(
    signal, fs: int, refractory_s: float = 0.2, threshold_frac: float = 0.3
) -> np.ndarray:
    """Lightweight Pan-Tompkins-style detector: derivative -> square -> moving
    integration -> peak pick with a refractory period. For the online path only;
    training uses ground-truth annotations."""
    x = np.asarray(signal, dtype=np.float64)
    diff = np.diff(x, prepend=x[0])
    squared = diff**2
    win = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    # Round away convolution floating-point noise: a wide flat-topped QRS energy
    # plateau can otherwise appear as several near-equal samples rather than one
    # exact plateau, which fragments scipy's plateau-midpoint peak picking and
    # biases the detected index toward the plateau's leading edge.
    integrated = np.round(integrated, decimals=12)
    peak_max = integrated.max()
    if peak_max <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(
        integrated,
        height=threshold_frac * peak_max,
        distance=max(1, int(refractory_s * fs)),
    )
    return peaks
