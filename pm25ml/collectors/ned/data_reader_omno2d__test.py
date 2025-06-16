import arrow
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr
from pm25ml.collectors.ned.coord_types import Lat, Lon
from pm25ml.collectors.ned.data_reader_omno2d import Omno2dReader
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.data_readers import NedDayData
from fsspec.spec import AbstractBufferedFile

PHONY_LAT_ADJUSTMENT = 90
PHONY_LON_ADJUSTMENT = 180

LON_RESOLUTION = 1.0
LAT_RESOLUTION = 1.0


@pytest.fixture
def fake_dataset_descriptor():
    """Mock dataset descriptor."""
    descriptor = NedDatasetDescriptor(
        dataset_name="fake_name",
        dataset_version="fake_version",
        start_date=arrow.get("2023-01-01"),
        end_date=arrow.get("2023-01-03"),
        filter_bounds=(Lon(-10.0), Lat(-10.0), Lon(10.0), Lat(10.0)),
        source_variable_name="mock_var",
        target_variable_name="new_mock_var",
    )
    return descriptor


@pytest.fixture
def fake_file_attributes():
    """Fake file attributes dataset."""
    dataset = xr.Dataset()
    dataset.attrs = {
        "GranuleYear": np.int32(2025),
        "GranuleMonth": np.int32(6),
        "GranuleDay": np.int32(15),
    }
    return dataset


@pytest.fixture
def fake_grid_info():
    """Fake grid info dataset."""
    dataset = xr.Dataset()
    dataset.attrs = {
        "GridSpan": "[-180, 180, -90, 90]",
        "GridSpacing": f"[{LON_RESOLUTION}, {LAT_RESOLUTION}]",
        "NumberOfLatitudesInGrid": np.int32(180),
        "NumberOfLongitudesInGrid": np.int32(360),
    }
    return dataset


@pytest.fixture
def fake_data_fields():
    """Fake data fields dataset."""
    data = np.random.rand(180, 360)
    dataset = xr.DataArray(
        data,
        dims=["phony_dim_0", "phony_dim_1"],
        # The phony dimensions are created by xarray when reading HDF5 files.
        # We just need to make sure they're the right size.
        # phony_dim_0 maps to lat (-90 to 90)
        # phony_dim_1 maps to lon (-180 to 180)
        coords={"phony_dim_0": np.arange(0, 180), "phony_dim_1": np.arange(0, 360)},
    ).to_dataset(name="mock_var")
    return dataset


def test_extract_data(
    fake_dataset_descriptor, fake_file_attributes, fake_grid_info, fake_data_fields
):
    """Test extracting data from OMI NO2 file."""
    mock_file = MagicMock(spec=AbstractBufferedFile)
    reader = Omno2dReader()

    def mock_open_dataset(file, group):
        if group == "HDFEOS/ADDITIONAL/FILE_ATTRIBUTES":
            return fake_file_attributes
        elif group == "HDFEOS/GRIDS/ColumnAmountNO2":
            return fake_grid_info
        elif group == "HDFEOS/GRIDS/ColumnAmountNO2/Data Fields":
            return fake_data_fields
        else:
            raise ValueError(f"Unexpected group: {group}")

    with patch("xarray.open_dataset", side_effect=mock_open_dataset):
        result = reader.extract_data(mock_file, fake_dataset_descriptor)

    assert isinstance(result, NedDayData)
    assert result.data.dims == ("lat", "lon")
    assert len(result.data.coords["lat"]) == 20  # Filtered bounds
    assert len(result.data.coords["lon"]) == 20  # Filtered bounds

    expected_mean = (
        fake_data_fields["mock_var"]
        .sel(
            phony_dim_0=slice(
                PHONY_LAT_ADJUSTMENT - 10 - LAT_RESOLUTION / 2,
                PHONY_LAT_ADJUSTMENT + 10 - LAT_RESOLUTION / 2,
            ),
            phony_dim_1=slice(
                PHONY_LON_ADJUSTMENT - 10 - LON_RESOLUTION / 2,
                PHONY_LON_ADJUSTMENT + 10 - LON_RESOLUTION / 2,
            ),
        )
        .mean()
        .item()
    )
    actual_mean = result.data.mean().item()
    assert actual_mean == expected_mean
