"""Low-timestep, multi-channel beat encoding.

Composes the delta (level-crossing) primitive into a compact spike-count
representation: encode each derivative order of the beat, pool the crossings
into a small number of timesteps, and stack the channels. This cuts the SNN's
sequential time loop from one step per ECG sample (256) down to `n_timesteps`
(e.g. 16-32), which is where the training-speed and energy wins come from, and
optionally adds derivative channels to expose more waveform shape to the network.
"""

import numpy as np

from neurocardio.encoding.delta import delta_encode


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    if std < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / std


def _derivative(x: np.ndarray, order: int) -> np.ndarray:
    """order-th discrete derivative, length-preserving (order 0 returns x)."""
    out = np.asarray(x, dtype=np.float64)
    for _ in range(order):
        out = np.diff(out, prepend=out[:1])
    return out


def pool_spikes(spikes: np.ndarray, n_timesteps: int) -> np.ndarray:
    """Sum up/down crossings into `n_timesteps` contiguous bins along time.

    Input [L, 2], output [n_timesteps, 2] of float32 counts. Pooling only
    regroups events in time, so the total count is conserved. Bins differ by at
    most one sample when L is not divisible by n_timesteps.
    """
    chunks = np.array_split(np.asarray(spikes, dtype=np.float32), n_timesteps, axis=0)
    pooled = np.stack([c.sum(axis=0) for c in chunks], axis=0)
    return pooled.astype(np.float32)


def encode_beat(signal, threshold: float, n_timesteps: int, derivative_orders=(0,)) -> np.ndarray:
    """Encode a beat into [n_timesteps, 2 * len(derivative_orders)] spike counts.

    For each derivative order, the (length-preserving) derivative is normalized
    so a single threshold is meaningful across orders, delta-encoded into up/down
    crossings, and pooled into `n_timesteps` bins. Channels are concatenated as
    [up_0, down_0, up_1, down_1, ...].
    """
    sig = np.asarray(signal, dtype=np.float64)
    channels = []
    for order in derivative_orders:
        deriv = _normalize(_derivative(sig, order))
        pooled = pool_spikes(delta_encode(deriv, threshold), n_timesteps)
        channels.append(pooled)
    return np.concatenate(channels, axis=1).astype(np.float32)
