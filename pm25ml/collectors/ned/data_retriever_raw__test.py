import pytest
from unittest.mock import patch, MagicMock
from pm25ml.collectors.ned.data_retriever_raw import RawEarthAccessDataRetriever
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.errors import NedMissingDataError
import arrow
from pm25ml.collectors.ned.coord_types import Lon, Lat


@pytest.fixture
def mock_dataset_descriptor():
    return NedDatasetDescriptor(
        dataset_name="mock_dataset",
        dataset_version="1",
        start_date=arrow.get("2025-01-01"),
        end_date=arrow.get("2025-01-31"),
        filter_bounds=(Lon(70.0), Lat(8.0), Lon(90.0), Lat(37.0)),
        variable_mapping={
            "mock_var": "mock_target_var",
        },
    )


@pytest.fixture
def mock_correct_granules():
    return [MagicMock()] * 31


@patch("pm25ml.collectors.ned.data_retriever_raw.earthaccess.search_data")
@patch("pm25ml.collectors.ned.data_retriever_raw.earthaccess.open")
def test__stream_files__valid_granules__returns_files(
    mock_open, mock_search_data, mock_dataset_descriptor, mock_correct_granules
):
    mock_search_data.return_value = mock_correct_granules
    mock_file = MagicMock()
    mock_open.return_value = [mock_file]

    retriever = RawEarthAccessDataRetriever()
    files = list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))

    assert len(files) == 31
    assert all(file is mock_file for file in files)


@patch("pm25ml.collectors.ned.data_retriever_raw.earthaccess.search_data")
def test__stream_files__no_granules__raises_error(mock_search_data, mock_dataset_descriptor):
    mock_search_data.return_value = []

    retriever = RawEarthAccessDataRetriever()

    with pytest.raises(NedMissingDataError):
        list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))


@patch("pm25ml.collectors.ned.data_retriever_raw.earthaccess.search_data")
def test__stream_files__granule_count_mismatch__raises_error(
    mock_search_data, mock_dataset_descriptor
):
    mock_search_data.return_value = [MagicMock()] * 15

    retriever = RawEarthAccessDataRetriever()

    with pytest.raises(NedMissingDataError):
        list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))


@patch("pm25ml.collectors.ned.data_retriever_raw.earthaccess.search_data")
def test__search_data__called_with_correct_arguments(
    mock_search_data, mock_dataset_descriptor, mock_correct_granules
):
    mock_search_data.return_value = mock_correct_granules
    retriever = RawEarthAccessDataRetriever()

    with (
        patch("pm25ml.collectors.ned.data_retriever_raw.logger.info"),
        patch(
            "pm25ml.collectors.ned.data_retriever_raw.earthaccess.open", return_value=[MagicMock()]
        ),
    ):
        list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))

    mock_search_data.assert_called_once_with(
        short_name=mock_dataset_descriptor.dataset_name,
        temporal=(
            mock_dataset_descriptor.start_date.format("YYYY-MM-DD"),
            mock_dataset_descriptor.end_date.format("YYYY-MM-DD"),
        ),
        count=-1,
        version=mock_dataset_descriptor.dataset_version,
    )
