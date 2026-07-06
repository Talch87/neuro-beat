import pytest

from neurocardio.data.splits import (
    DS1_RECORDS, DS2_RECORDS, PACED_RECORDS, get_split,
)


def test_ds1_ds2_sizes():
    assert len(DS1_RECORDS) == 22
    assert len(DS2_RECORDS) == 22


def test_ds1_ds2_disjoint():
    assert set(DS1_RECORDS).isdisjoint(set(DS2_RECORDS))


def test_paced_records_excluded_from_both():
    both = set(DS1_RECORDS) | set(DS2_RECORDS)
    assert both.isdisjoint(set(PACED_RECORDS))
    assert set(PACED_RECORDS) == {"102", "104", "107", "217"}


def test_get_split_names():
    assert get_split("train") == DS1_RECORDS
    assert get_split("test") == DS2_RECORDS
    with pytest.raises(ValueError):
        get_split("bogus")
