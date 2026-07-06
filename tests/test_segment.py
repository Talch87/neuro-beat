import numpy as np

from neurocardio.data.segment import AAMI_CLASSES, segment_beats, symbol_to_aami


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
