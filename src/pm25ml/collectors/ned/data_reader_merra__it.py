"""
Integration test for MerraDataReader.
"""

import pytest
from pathlib import Path
from pm25ml.collectors.ned.data_reader_merra import MerraDataReader
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.coord_types import Lon, Lat
import arrow

pytestmark = pytest.mark.integration


@pytest.fixture
def example_file_path() -> Path:
    """
    Fixture to provide the path to the example MERRA-2 file.
    """
    return (
        Path(__file__).parent
        / "data_reader_merra__it_assets"
        / "M2T1NXAER.5.12.4_MERRA2_400.tavg1_2d_aer_Nx.20230101_TOTEXTTAU_subsetted.nc4"
    )


@pytest.fixture
def dataset_descriptor() -> NedDatasetDescriptor:
    """
    Fixture to provide a dataset descriptor for the test.
    """
    return NedDatasetDescriptor(
        dataset_name="MERRA2_AER",
        dataset_version="v5.12.4",
        start_date=arrow.get("2023-01-01"),
        end_date=arrow.get("2023-01-01"),
        filter_bounds=(Lon(70.0), Lat(10.0), Lon(90.0), Lat(30.0)),
        variable_mapping={
            "TOTEXTTAU": "AerosolOpticalDepth",
        },
        level=None,
    )


def test__merra_data_reader__read_example_file__extracts_data(
    example_file_path, dataset_descriptor
):
    """
    Test that MerraDataReader can read an example file and extract data.
    """
    reader = MerraDataReader()

    with example_file_path.open("rb") as file:
        ned_day_data = reader.extract_data(file, dataset_descriptor)

        assert ned_day_data.data is not None
        assert dict(ned_day_data.data.dims) == {
            "lat": 41,
            "lon": 33,
        }
        assert ned_day_data.data.coords["lon"].min() == 70.0
        assert ned_day_data.data.coords["lon"].max() == 90.0
        assert ned_day_data.data.coords["lat"].min() == 10.0
        assert ned_day_data.data.coords["lat"].max() == 30.0
        assert ned_day_data.date == "2023-01-01"
