import numpy as np
import wfdb

from neurocardio.data.records import Record, load_record


def _write_fixture(dirpath):
    fs = 360
    sig = np.zeros((fs, 1), dtype=np.float64)
    sig[100, 0] = 1.0
    sig[250, 0] = 1.0
    wfdb.wrsamp(
        "rec1", fs=fs, units=["mV"], sig_name=["MLII"],
        p_signal=sig, write_dir=str(dirpath),
    )
    wfdb.wrann(
        "rec1", "atr",
        sample=np.array([100, 250]),
        symbol=["N", "V"],
        write_dir=str(dirpath),
    )


def test_load_record_returns_signal_fs_and_annotations(tmp_path):
    _write_fixture(tmp_path)
    rec = load_record(tmp_path, "rec1", lead_index=0)
    assert isinstance(rec, Record)
    assert rec.fs == 360
    assert rec.signal.shape == (360,)
    assert list(rec.ann_samples) == [100, 250]
    assert rec.ann_symbols == ["N", "V"]
