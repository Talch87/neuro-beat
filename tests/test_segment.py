import numpy as np

from neurocardio.data.segment import (
    AAMI_CLASSES,
    beat_rr_features,
    segment_beats,
    symbol_to_aami,
)


def test_rr_features_align_with_segment_beats():
    # 5 beats at samples 300,600,900,1200,1500 (RR=300), one premature at 1650.
    signal = np.zeros(2000)
    ann_samples = np.array([300, 600, 900, 1200, 1500, 1650])
    ann_symbols = ["N", "N", "N", "N", "N", "A"]  # last is a premature SVEB
    beats, labels = segment_beats(signal, ann_samples, ann_symbols, 128, 128)
    rr = beat_rr_features(ann_samples, ann_symbols, len(signal), 128, 128)
    assert rr.shape == (len(beats), 3)  # one RR row per kept beat
    # median RR is 300; a regular beat has pre/median ~= 1.0
    assert abs(rr[1, 0] - 1.0) < 1e-9
    # the premature beat (1650, pre_RR=150) has pre/median = 0.5 < 1 -- the timing cue
    assert rr[-1, 0] < 0.6


def test_rr_features_empty_when_no_beats():
    rr = beat_rr_features(np.array([10]), ["+"], 1000, 128, 128)
    assert rr.shape == (0, 3)


def test_aami_class_order_is_fixed():
    assert AAMI_CLASSES == ["N", "SVEB", "VEB", "F", "Q"]


def test_symbol_mapping():
    assert symbol_to_aami("N") == "N"
    assert symbol_to_aami("L") == "N"
    assert symbol_to_aami("A") == "SVEB"
    assert symbol_to_aami("V") == "VEB"
    assert symbol_to_aami("E") == "VEB"
    assert symbol_to_aami("F") == "F"
    assert symbol_to_aami("/") == "Q"
    assert symbol_to_aami("+") is None  # rhythm marker, not a beat


def test_segment_windows_and_labels():
    signal = np.arange(2000, dtype=np.float64)
    ann_samples = np.array([500, 1000, 1500])
    ann_symbols = ["N", "V", "A"]
    beats, labels = segment_beats(
        signal, ann_samples, ann_symbols, window_before=128, window_after=128
    )
    assert beats.shape == (3, 256)
    assert beats[0, 128] == 500.0
    assert list(labels) == [0, 2, 1]  # N=0, VEB=2, SVEB=1


def test_segment_drops_edge_and_nonbeat_annotations():
    signal = np.zeros(1000, dtype=np.float64)
    ann_samples = np.array([10, 500, 995])  # 10 and 995 too close to edges
    ann_symbols = ["N", "+", "N"]  # 500 is a non-beat marker
    beats, labels = segment_beats(
        signal, ann_samples, ann_symbols, window_before=128, window_after=128
    )
    assert beats.shape[0] == 0  # all dropped
