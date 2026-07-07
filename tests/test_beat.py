import numpy as np

from neurocardio.encoding.beat import encode_beat, pool_spikes
from neurocardio.encoding.delta import delta_encode


def test_pool_spikes_shape_and_counts():
    spikes = np.zeros((256, 2), dtype=np.float32)
    spikes[0:3, 0] = 1.0  # 3 up crossings in the first bin
    spikes[8:10, 1] = 1.0  # 2 down crossings in the second bin
    pooled = pool_spikes(spikes, n_timesteps=32)  # 256 / 32 = 8 samples per bin
    assert pooled.shape == (32, 2)
    assert pooled[0, 0] == 3.0  # first bin summed the 3 up crossings
    assert pooled[1, 1] == 2.0  # second bin summed the 2 down crossings


def test_pool_conserves_total_events():
    rng = np.random.default_rng(0)
    spikes = (rng.random((256, 2)) < 0.1).astype(np.float32)
    pooled = pool_spikes(spikes, n_timesteps=16)
    assert pooled.shape == (16, 2)
    assert pooled.sum() == spikes.sum()  # pooling only regroups, never drops events


def test_pool_handles_indivisible_timesteps():
    spikes = np.ones((10, 2), dtype=np.float32)
    pooled = pool_spikes(spikes, n_timesteps=3)  # 10 not divisible by 3
    assert pooled.shape == (3, 2)
    assert pooled.sum() == 20.0  # all events preserved


def test_encode_beat_channel_count_scales_with_derivatives():
    sig = np.cumsum(np.random.default_rng(1).standard_normal(256)) * 0.02
    x0 = encode_beat(sig, threshold=0.1, n_timesteps=32, derivative_orders=[0])
    x1 = encode_beat(sig, threshold=0.1, n_timesteps=32, derivative_orders=[0, 1])
    x2 = encode_beat(sig, threshold=0.1, n_timesteps=32, derivative_orders=[0, 1, 2])
    assert x0.shape == (32, 2)
    assert x1.shape == (32, 4)
    assert x2.shape == (32, 6)
    assert x0.dtype == np.float32


def test_encode_beat_raw_channel_matches_pooled_delta():
    # order 0 with full-length pooling should reproduce the delta encoding of the
    # normalized signal (encode_beat normalizes each channel internally).
    rng = np.random.default_rng(2)
    sig = np.cumsum(rng.standard_normal(256)) * 0.02
    norm = (sig - sig.mean()) / sig.std()
    expected = delta_encode(norm, threshold=0.1)  # [256, 2]
    got = encode_beat(sig, threshold=0.1, n_timesteps=256, derivative_orders=[0])
    assert got.shape == (256, 2)
    assert np.allclose(got, expected)
