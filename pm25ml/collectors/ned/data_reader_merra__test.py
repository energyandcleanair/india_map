import pytest
from unittest.mock import MagicMock, patch
from pm25ml.collectors.ned.data_reader_merra import MerraDataReader
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.data_readers import NedDayData
import xarray as xr
import numpy as np
from fsspec.spec import AbstractBufferedFile


@pytest.fixture
def mock_dataset_descriptor():
    """Mock dataset descriptor."""
    descriptor = MagicMock(spec=NedDatasetDescriptor)
    descriptor.source_variable_name = "mock_var"
    descriptor.filter_bounds = [-10, -10, 10, 10]
    return descriptor


@pytest.fixture
def dataset_with_all_dimensions_including_lev_down():
    data = np.arange(18 * 36 * 5 * 3).reshape(18, 36, 5, 3)  # lat, lon, time, lev

    dataset = xr.DataArray(
        data,
        dims=["lat", "lon", "time", "lev"],
        coords={
            "lat": np.arange(-90, 90, 10),
            "lon": np.arange(-180, 180, 10),
            "time": np.arange(5),
            "lev": np.arange(3),
        },
    ).to_dataset(name="mock_var")
    dataset.attrs["RangeBeginningDate"] = "2025-06-01"
    dataset["lev"].attrs["positive"] = "down"
    return dataset


@pytest.fixture
def dataset_with_all_dimensions_including_lev_up():
    data = np.arange(18 * 36 * 5 * 3).reshape(18, 36, 5, 3)
    dataset = xr.DataArray(
        data,
        dims=["lat", "lon", "time", "lev"],
        coords={
            "lat": np.arange(-90, 90, 10),
            "lon": np.arange(-180, 180, 10),
            "time": np.arange(5),
            "lev": np.arange(3),
        },
    ).to_dataset(name="mock_var")
    dataset.attrs["RangeBeginningDate"] = "2025-06-01"
    dataset["lev"].attrs["positive"] = "up"
    return dataset


@pytest.fixture
def dataset_missing_lev_dimension():
    """Fixture for a dataset with 3 dimensions to reduce to 2."""
    data = np.arange(18 * 36 * 5).reshape(18, 36, 5)  # lat, lon, time
    dataset = xr.DataArray(
        data,
        dims=["lat", "lon", "time"],
        coords={
            "lat": np.arange(-90, 90, 10),
            "lon": np.arange(-180, 180, 10),
            "time": np.arange(5),
        },
    ).to_dataset(name="mock_var")
    dataset.attrs["RangeBeginningDate"] = "2025-06-01"
    return dataset


@pytest.fixture
def dataset_missing_time_dimension():
    """Fixture for a dataset with multiple levels."""
    data = np.arange(18 * 36 * 3).reshape(18, 36, 3)  # lat, lon, lev
    dataset = xr.DataArray(
        data,
        dims=["lat", "lon", "lev"],
        coords={
            "lat": np.arange(-90, 90, 10),
            "lon": np.arange(-180, 180, 10),
            "lev": np.arange(3),
        },
    ).to_dataset(name="mock_var")
    dataset["lev"].attrs["positive"] = "down"
    dataset.attrs["RangeBeginningDate"] = "2025-06-01"
    return dataset


@pytest.fixture
def dataset_with_unexpected_dimensions():
    """Fixture for a dataset with unexpected dimensions."""
    data = np.arange(18 * 36 * 5 * 3).reshape(18, 36, 5, 3)
    dataset = xr.DataArray(
        data,
        dims=["lat", "lon", "time", "unexpected"],
        coords={
            "lat": np.arange(-90, 90, 10),
            "lon": np.arange(-180, 180, 10),
            "time": np.arange(5),
            "unexpected": np.arange(3),
        },
    ).to_dataset(name="mock_var")
    dataset.attrs["RangeBeginningDate"] = "2025-06-01"
    return dataset


def test__MerraDataReader_extract_data__correct_file_passed(
    mock_dataset_descriptor, dataset_with_all_dimensions_including_lev_down
):
    """Test extracting data with a correct file."""
    mock_file = MagicMock(spec=AbstractBufferedFile)
    reader = MerraDataReader()
    with patch(
        "xarray.open_dataset", return_value=dataset_with_all_dimensions_including_lev_down
    ) as mock_open_dataset:
        reader.extract_data(mock_file, mock_dataset_descriptor)

        mock_open_dataset.assert_called_once_with(mock_file, chunks="auto", engine="h5netcdf")


def test__MerraDataReader_extract_data__valid_with_date__date_correctly_extracted(
    mock_dataset_descriptor, dataset_with_all_dimensions_including_lev_down
):
    """Test extracting data with valid inputs."""
    mock_file = MagicMock(spec=AbstractBufferedFile)
    with patch("xarray.open_dataset", return_value=dataset_with_all_dimensions_including_lev_down):
        reader = MerraDataReader()
        result = reader.extract_data(mock_file, mock_dataset_descriptor)

    assert isinstance(result, NedDayData)
    assert result.date == "2025-06-01"


def test__MerraDataReader_extract_data__valid_missing_date__raises_value_error(
    mock_dataset_descriptor, dataset_with_all_dimensions_including_lev_down
):
    """Test extracting data when the date attribute is missing."""
    dataset_with_all_dimensions_including_lev_down.attrs.pop("RangeBeginningDate", None)
    mock_file = MagicMock(spec=AbstractBufferedFile)
    with patch("xarray.open_dataset", return_value=dataset_with_all_dimensions_including_lev_down):
        reader = MerraDataReader()
        with pytest.raises(
            ValueError, match="Dataset does not contain a valid 'RangeBeginningDate' attribute."
        ):
            reader.extract_data(mock_file, mock_dataset_descriptor)


def test__MerraDataReader_extract_data__valid_lev_positive_down__data_correctly_filtered(
    mock_dataset_descriptor, dataset_with_all_dimensions_including_lev_down
):
    """Test extracting data with valid inputs."""
    mock_file = MagicMock(spec=AbstractBufferedFile)
    with patch("xarray.open_dataset", return_value=dataset_with_all_dimensions_including_lev_down):
        reader = MerraDataReader()
        result = reader.extract_data(mock_file, mock_dataset_descriptor)

    assert isinstance(result, NedDayData)
    assert result.data.dims == ("lat", "lon")
    # Check that the data is filtered correctly to the right bounds
    assert len(result.data.coords["lat"]) == 3
    assert len(result.data.coords["lon"]) == 3

    # Calculate the expected mean value directly for the dataset
    expected_mean = (
        dataset_with_all_dimensions_including_lev_down["mock_var"]
        .isel(lev=-1)
        .sel(lat=slice(-10, 10), lon=slice(-10, 10))
        .mean()
        .item()
    )
    actual_mean = result.data.mean().item()
    assert actual_mean == expected_mean


def test__MerraDataReader_extract_data__valid_lev_positive_up__data_correctly_filtered(
    mock_dataset_descriptor, dataset_with_all_dimensions_including_lev_up
):
    """Test extracting data with valid inputs."""
    mock_file = MagicMock(spec=AbstractBufferedFile)
    with patch("xarray.open_dataset", return_value=dataset_with_all_dimensions_including_lev_up):
        reader = MerraDataReader()
        result = reader.extract_data(mock_file, mock_dataset_descriptor)

    assert isinstance(result, NedDayData)
    assert result.data.dims == ("lat", "lon")
    # Check that the data is filtered correctly to the right bounds
    assert len(result.data.coords["lat"]) == 3
    assert len(result.data.coords["lon"]) == 3

    expected_mean = (
        dataset_with_all_dimensions_including_lev_up["mock_var"]
        .isel(lev=0)
        .sel(lat=slice(-10, 10), lon=slice(-10, 10))
        .mean()
        .item()
    )
    actual_mean = result.data.mean().item()
    assert actual_mean == expected_mean


def test__MerraDataReader_extract_data__valid_without_lev_dimension__data_correctly_filtered(
    mock_dataset_descriptor, dataset_missing_lev_dimension
):
    """Test extracting data when the 'lev' dimension is missing."""
    mock_file = MagicMock(spec=AbstractBufferedFile)
    with patch("xarray.open_dataset", return_value=dataset_missing_lev_dimension):
        reader = MerraDataReader()
        result = reader.extract_data(mock_file, mock_dataset_descriptor)

    assert isinstance(result, NedDayData)
    assert result.date == "2025-06-01"
    assert result.data.dims == ("lat", "lon")
    # Check that the data is filtered correctly to the right bounds
    assert len(result.data.coords["lat"]) == 3
    assert len(result.data.coords["lon"]) == 3

    expected_mean = (
        dataset_missing_lev_dimension["mock_var"]
        .sel(lat=slice(-10, 10), lon=slice(-10, 10))
        .mean()
        .item()
    )
    actual_mean = result.data.mean().item()
    assert actual_mean == expected_mean


def test__MerraDataReader_extract_data__invalid_dimensions__raises_value_error(
    mock_dataset_descriptor, dataset_with_unexpected_dimensions
):
    """Test extracting data when the data array has invalid dimensions."""
    mock_file = MagicMock(spec=AbstractBufferedFile)
    with patch("xarray.open_dataset", return_value=dataset_with_unexpected_dimensions):
        reader = MerraDataReader()
        with pytest.raises(
            ValueError,
            match=(
                "Dataset contains unexpected dimensions: \\['unexpected'\\]. "
                "Actual dimensions are: \\['lat', 'lon', 'time', 'unexpected'\\]. "
                "Allowable dimensions are: \\['lon', 'lat', 'time', 'lev'\\]."
            ),
        ):
            reader.extract_data(mock_file, mock_dataset_descriptor)


def test__MerraDataReader_extract_data__missing_time__raises_value_error(
    mock_dataset_descriptor, dataset_missing_time_dimension
):
    """Test extracting data when the data array has a 'lev' dimension."""
    mock_file = MagicMock(spec=AbstractBufferedFile)
    with patch("xarray.open_dataset", return_value=dataset_missing_time_dimension):
        reader = MerraDataReader()
        with pytest.raises(
            ValueError,
            match=(
                "Dataset is missing expected dimensions: \\['time'\\]. "
                "Actual dimensions are: \\['lat', 'lon', 'lev'\\]. "
                "Required dimensions are: \\['lon', 'lat', 'time'\\]."
            ),
        ):
            reader.extract_data(mock_file, mock_dataset_descriptor)
