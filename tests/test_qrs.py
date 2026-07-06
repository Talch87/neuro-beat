import numpy as np

from neurocardio.stream.qrs import find_r_peaks


def test_finds_peaks_near_known_positions():
    fs = 360
    n = fs * 5
    sig = np.zeros(n)
    true_peaks = [300, 700, 1100, 1500]
    for p in true_peaks:
        sig[p - 2 : p + 3] += np.array([0.2, 0.6, 1.0, 0.6, 0.2])
    peaks = find_r_peaks(sig, fs=fs)
    assert len(peaks) == len(true_peaks)
    for detected, expected in zip(sorted(peaks), true_peaks):
        assert abs(int(detected) - expected) <= 5


def test_refractory_prevents_double_counting():
    fs = 360
    sig = np.zeros(fs)
    sig[100:103] = [0.5, 1.0, 0.5]
    sig[105:108] = [0.5, 1.0, 0.5]  # 5 samples later, within refractory
    peaks = find_r_peaks(sig, fs=fs, refractory_s=0.2)
    assert len(peaks) == 1
