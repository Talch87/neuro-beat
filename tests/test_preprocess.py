import numpy as np

from neurocardio.data.preprocess import bandpass_filter, normalize


def test_bandpass_removes_dc_offset():
    fs = 360
    t = np.arange(fs * 2) / fs
    clean = np.sin(2 * np.pi * 10 * t)      # 10 Hz, in-band
    with_dc = clean + 5.0                    # DC offset (0 Hz, out-of-band)
    out = bandpass_filter(with_dc, fs=fs, low=0.5, high=40.0, order=4)
    assert abs(out.mean()) < 0.05            # DC essentially removed


def test_bandpass_attenuates_out_of_band():
    fs = 360
    t = np.arange(fs * 2) / fs
    inband = np.sin(2 * np.pi * 10 * t)
    highfreq = np.sin(2 * np.pi * 120 * t)   # above 40 Hz cutoff
    out = bandpass_filter(inband + highfreq, fs=fs, low=0.5, high=40.0, order=4)
    assert out.std() < (inband + highfreq).std()


def test_normalize_zero_mean_unit_std():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = normalize(x)
    assert abs(out.mean()) < 1e-9
    assert abs(out.std() - 1.0) < 1e-6


def test_normalize_constant_signal_is_safe():
    out = normalize(np.ones(10))
    assert np.all(np.isfinite(out))
