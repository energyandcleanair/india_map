import pytest
from pm25ml.hive_path import HivePath


def test__init__with_valid_path__extracts_metadata():
    path = "country=india/dataset=era5_land/month=2023-01"
    hive_path = HivePath(path)
    assert hive_path.metadata == {"country": "india", "dataset": "era5_land", "month": "2023-01"}


def test__extract_metadata_with_no_metadata__returns_empty_dict():
    path = "no_metadata_here"
    hive_path = HivePath(path)
    assert hive_path.metadata == {}


def test__require_key_with_existing_key__returns_value():
    path = "country=india/dataset=era5_land_month=2023-01"
    hive_path = HivePath(path)
    assert hive_path.require_key("country") == "india"


def test__require_key_with_missing_key__raises_value_error():
    path = "country=india/dataset=era5_land_month=2023-01"
    hive_path = HivePath(path)
    with pytest.raises(
        ValueError,
        match="Expected 'missing_key' key in country=india/dataset=era5_land_month=2023-01, but it is missing.",
    ):
        hive_path.require_key("missing_key")
