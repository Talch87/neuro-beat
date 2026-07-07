import numpy as np
import wfdb

from neurocardio.data.records import Record, list_records, load_record


def _write_fixture(dirpath):
    fs = 360
    sig = np.zeros((fs, 1), dtype=np.float64)
    sig[100, 0] = 1.0
    sig[250, 0] = 1.0
    wfdb.wrsamp(
        "rec1",
        fs=fs,
        units=["mV"],
        sig_name=["MLII"],
        p_signal=sig,
        write_dir=str(dirpath),
    )
    wfdb.wrann(
        "rec1",
        "atr",
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


def _write_fixture_at(dirpath, fs, sample, symbol="N", record_id="ext1"):
    sig = np.zeros((fs, 1), dtype=np.float64)  # 1 second
    sig[sample, 0] = 1.0
    wfdb.wrsamp(
        record_id, fs=fs, units=["mV"], sig_name=["II"], p_signal=sig, write_dir=str(dirpath)
    )
    wfdb.wrann(
        record_id, "atr", sample=np.array([sample]), symbol=[symbol], write_dir=str(dirpath)
    )


def test_load_record_resamples_signal_and_rescales_annotations(tmp_path):
    # A 128 Hz record with a beat at 0.5 s (sample 64) -> 360 Hz grid, sample 180.
    _write_fixture_at(tmp_path, fs=128, sample=64)
    rec = load_record(tmp_path, "ext1", lead_index=0, target_fs=360)
    assert rec.fs == 360
    assert abs(rec.signal.shape[0] - 360) <= 1
    assert rec.ann_samples.tolist() == [180]  # round(64 * 360 / 128)


def test_list_records_prefers_records_index(tmp_path):
    (tmp_path / "RECORDS").write_text("800\n801\n802\n")
    assert list_records(tmp_path) == ["800", "801", "802"]


def test_list_records_falls_back_to_header_glob(tmp_path):
    (tmp_path / "b.hea").write_text("")
    (tmp_path / "a.hea").write_text("")
    assert list_records(tmp_path) == ["a", "b"]
