import inspect

import pytest

from neurocardio.data import download


def test_download_functions_exist_and_signatures():
    assert callable(download.download_mitdb)
    assert callable(download.download_ptbdb)
    sig = inspect.signature(download.download_mitdb)
    assert "dest" in sig.parameters


@pytest.mark.slow
def test_download_mitdb_real(tmp_path):
    from neurocardio.data.download import download_mitdb

    out = download_mitdb(tmp_path / "mitdb")
    assert (out / "100.dat").exists()
