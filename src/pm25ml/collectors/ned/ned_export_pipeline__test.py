import pytest
from arrow import Arrow
from fsspec.implementations.memory import MemoryFileSystem
from unittest.mock import MagicMock
import polars as pl
import xarray as xr
from polars.testing import assert_frame_equal
from polars import DataFrame

from pm25ml.collectors.grid_loader import Grid
from pm25ml.collectors.ned.coord_types import Lat, Lon
from pm25ml.collectors.ned.data_readers import NedDataReader, NedDayData
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.ned_export_pipeline import NedExportPipeline, NedPipelineConstructor
from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.ned.data_retriever_raw import RawEarthAccessDataRetriever


@pytest.fixture
def mock_archive_storage():
    filesystem = MemoryFileSystem()
    destination_bucket = "mock_bucket"
    return IngestArchiveStorage(filesystem=filesystem, destination_bucket=destination_bucket)


@pytest.fixture
def mock_dataset_descriptor():
    return NedDatasetDescriptor(
        dataset_name="mock_dataset",
        dataset_version="1.0",
        start_date=Arrow(2025, 1, 1),
        end_date=Arrow(2025, 1, 31),
        filter_bounds=(Lon(70.0), Lat(8.0), Lon(90.0), Lat(37.0)),
        variable_mapping={"mock_var": "mock_target_var"},
    )


@pytest.fixture
def mock_grid():
    return Grid(
        DataFrame(
            {
                "grid_id": ["01", "02", "03", "04"],
                "lon": [0.5, 1.5, 2.5, 3.5],
                "lat": [2.5, 3.5, 4.5, 5.5],
            }
        )
    )


def test__NedPipelineConstructor__valid_inputs__creates_pipeline(
    mock_archive_storage, mock_dataset_descriptor, mock_grid
):
    # Mock inputs
    mock_dataset_reader = NedDataReader()
    mock_result_subpath = "mock/subpath"

    # Create NedPipelineConstructor
    constructor = NedPipelineConstructor(
        archive_storage=mock_archive_storage,
        grid=mock_grid,
    )

    # Construct pipeline
    pipeline = constructor.construct(
        dataset_descriptor=mock_dataset_descriptor,
        dataset_reader=mock_dataset_reader,
        result_subpath=mock_result_subpath,
    )

    # Assertions
    assert isinstance(pipeline, NedExportPipeline)
    assert pipeline._grid is mock_grid
    assert pipeline.archive_storage is mock_archive_storage
    assert pipeline.dataset_descriptor is mock_dataset_descriptor
    assert pipeline.dataset_reader is mock_dataset_reader
    assert pipeline.result_subpath == mock_result_subpath


def test__NedPipelineConstructor__dataset_retriever_logic__uses_default_retriever(
    mock_grid,
):
    # Mock inputs
    mock_archive_storage = IngestArchiveStorage(
        filesystem=MemoryFileSystem(), destination_bucket="mock_bucket"
    )
    mock_dataset_descriptor = NedDatasetDescriptor(
        dataset_name="mock_dataset",
        dataset_version="1.0",
        start_date=Arrow(2025, 1, 1),
        end_date=Arrow(2025, 1, 31),
        filter_bounds=(Lon(70.0), Lat(8.0), Lon(90.0), Lat(37.0)),
        variable_mapping={"mock_var": "mock_target_var"},
    )
    mock_dataset_reader = MagicMock()
    mock_result_subpath = "mock/subpath"

    # Create NedPipelineConstructor
    constructor = NedPipelineConstructor(
        archive_storage=mock_archive_storage,
        grid=mock_grid,
    )

    # Construct pipeline without providing a dataset_retriever
    pipeline = constructor.construct(
        dataset_descriptor=mock_dataset_descriptor,
        dataset_reader=mock_dataset_reader,
        result_subpath=mock_result_subpath,
    )

    # Assertions
    assert isinstance(pipeline.dataset_retriever, RawEarthAccessDataRetriever)


def test__NedPipelineConstructor__dataset_retriever_logic__uses_provided_retriever(
    mock_grid,
):
    mock_archive_storage = IngestArchiveStorage(
        filesystem=MemoryFileSystem(), destination_bucket="mock_bucket"
    )
    mock_dataset_descriptor = NedDatasetDescriptor(
        dataset_name="mock_dataset",
        dataset_version="1.0",
        start_date=Arrow(2025, 1, 1),
        end_date=Arrow(2025, 1, 31),
        filter_bounds=(Lon(70.0), Lat(8.0), Lon(90.0), Lat(37.0)),
        variable_mapping={"mock_var": "mock_target_var"},
    )
    mock_dataset_reader = MagicMock()
    mock_result_subpath = "mock/subpath"
    mock_dataset_retriever = MagicMock()

    # Create NedPipelineConstructor
    constructor = NedPipelineConstructor(
        archive_storage=mock_archive_storage,
        grid=mock_grid,
    )

    # Construct pipeline with a provided dataset_retriever
    pipeline = constructor.construct(
        dataset_descriptor=mock_dataset_descriptor,
        dataset_reader=mock_dataset_reader,
        dataset_retriever=mock_dataset_retriever,
        result_subpath=mock_result_subpath,
    )

    # Assertions
    assert pipeline.dataset_retriever is mock_dataset_retriever


def test__NedExportPipeline__happy_path__regrids_data_correctly():
    # Create a mock grid (4x4 grid)
    mock_grid = Grid(
        DataFrame(
            {
                "grid_id": ["01", "02", "03", "04"],
                "lon": [0.5, 1.5, 2.5, 3.5],
                "lat": [2.5, 3.5, 4.5, 5.5],
            }
        )
    )

    # Create mock data (2x2 grid with larger gaps between lat and lon)
    mock_data = NedDayData(
        dataset=xr.Dataset(
            {
                "mock_var": (
                    ["lat", "lon"],
                    [
                        [1.0, 2.0, 3.0],
                        [3.0, 4.0, 5.0],
                        [5.0, 6.0, 7.0],
                    ],
                )
            },
            coords={"lat": [2.0, 4.0, 6.0], "lon": [0.0, 2.0, 4.0]},
        ),
        date="2025-06-01",
    )

    # Mock the dataset retriever
    mock_dataset_retriever = MagicMock()
    mock_dataset_retriever.stream_files.return_value = [MagicMock()]

    # Mock the dataset reader
    mock_dataset_reader = MagicMock()
    mock_dataset_reader.extract_data.return_value = mock_data

    # Mock the archive storage
    mock_archive_storage = MagicMock()
    mock_archive_storage.write_to_destination.side_effect = lambda table, result_subpath: table

    # Use a real dataset descriptor
    real_dataset_descriptor = NedDatasetDescriptor(
        dataset_name="real_dataset",
        dataset_version="1.0",
        start_date=Arrow(2025, 6, 1),
        end_date=Arrow(2025, 6, 1),
        filter_bounds=(Lon(70.0), Lat(8.0), Lon(90.0), Lat(37.0)),
        variable_mapping={"mock_var": "mock_var_target"},
    )

    # Create the pipeline
    pipeline = NedExportPipeline(
        grid=mock_grid,
        archive_storage=mock_archive_storage,
        dataset_descriptor=real_dataset_descriptor,
        dataset_retriever=mock_dataset_retriever,
        dataset_reader=mock_dataset_reader,
        result_subpath="mock/subpath",
    )

    # Run the upload method
    pipeline.upload()

    # Assertions
    mock_archive_storage.write_to_destination.assert_called_once()
    written_table = mock_archive_storage.write_to_destination.call_args[1]["table"]

    manually_computed_expected_values = [
        1.75,
        3.25,
        4.75,
        6.25,
    ]

    # Check that the regridded data matches the expected values
    expected_data = pl.DataFrame(
        {
            "grid_id": ["01", "02", "03", "04"],
            "date": ["2025-06-01"] * 4,
            "mock_var_target": manually_computed_expected_values,
        }
    )

    assert_frame_equal(
        written_table,
        expected_data,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        rtol=0,
        atol=0.01,
    )


def test__NedExportPipeline__export_result__matches_expected_format_and_values():
    # Create a mock grid (4x4 grid)
    mock_grid = Grid(
        DataFrame(
            {
                "grid_id": ["01", "02", "03", "04"],
                "lon": [0.5, 1.5, 2.5, 3.5],
                "lat": [2.5, 3.5, 4.5, 5.5],
            }
        )
    )

    # Create mock data (2x2 grid with larger gaps between lat and lon)
    mock_data = NedDayData(
        dataset=xr.Dataset(
            {
                "mock_var": (
                    ["lat", "lon"],
                    [
                        [1.0, 2.0, 3.0],
                        [3.0, 4.0, 5.0],
                        [5.0, 6.0, 7.0],
                    ],
                )
            },
            coords={"lat": [2.0, 4.0, 6.0], "lon": [0.0, 2.0, 4.0]},
        ),
        date="2025-06-01",
    )

    # Mock the dataset retriever
    mock_dataset_retriever = MagicMock()
    mock_dataset_retriever.stream_files.return_value = [MagicMock()]

    # Mock the dataset reader
    mock_dataset_reader = MagicMock()
    mock_dataset_reader.extract_data.return_value = mock_data

    # Mock the archive storage
    mock_archive_storage = MagicMock()
    mock_archive_storage.write_to_destination.side_effect = lambda table, result_subpath: table

    # Use a real dataset descriptor
    real_dataset_descriptor = NedDatasetDescriptor(
        dataset_name="real_dataset",
        dataset_version="1.0",
        start_date=Arrow(2025, 6, 1),
        end_date=Arrow(2025, 6, 1),
        filter_bounds=(Lon(70.0), Lat(8.0), Lon(90.0), Lat(37.0)),
        variable_mapping={
            "mock_var": "mock_var_target",
        },
    )

    # Create the pipeline
    pipeline = NedExportPipeline(
        grid=mock_grid,
        archive_storage=mock_archive_storage,
        dataset_descriptor=real_dataset_descriptor,
        dataset_retriever=mock_dataset_retriever,
        dataset_reader=mock_dataset_reader,
        result_subpath="mock/subpath",
    )

    # Run the upload method
    result = pipeline.get_config_metadata()

    assert result.result_subpath == "mock/subpath"
    assert result.id_columns == {"date", "grid_id"}
    assert result.value_columns == {"mock_var_target"}
    assert result.expected_rows == 4


def test__NedExportPipeline__multiple_variables__processes_correctly():
    # Create a mock grid (4x4 grid)
    mock_grid = Grid(
        DataFrame(
            {
                "grid_id": ["01", "02", "03", "04"],
                "lon": [0.5, 1.5, 2.5, 3.5],
                "lat": [2.5, 3.5, 4.5, 5.5],
            }
        )
    )

    # Create mock data with multiple variables
    mock_data = NedDayData(
        dataset=xr.Dataset(
            {
                "var1": (
                    ["lat", "lon"],
                    [
                        [1.0, 2.0, 3.0],
                        [3.0, 4.0, 5.0],
                        [5.0, 6.0, 7.0],
                    ],
                ),
                "var2": (
                    ["lat", "lon"],
                    [
                        [10.0, 20.0, 30.0],
                        [30.0, 40.0, 50.0],
                        [50.0, 60.0, 70.0],
                    ],
                ),
            },
            coords={"lat": [2.0, 4.0, 6.0], "lon": [0.0, 2.0, 4.0]},
        ),
        date="2025-06-01",
    )

    # Mock the dataset retriever
    mock_dataset_retriever = MagicMock()
    mock_dataset_retriever.stream_files.return_value = [MagicMock()]

    # Mock the dataset reader
    mock_dataset_reader = MagicMock()
    mock_dataset_reader.extract_data.return_value = mock_data

    # Mock the archive storage
    mock_archive_storage = MagicMock()
    mock_archive_storage.write_to_destination.side_effect = lambda table, result_subpath: table

    # Use a real dataset descriptor
    real_dataset_descriptor = NedDatasetDescriptor(
        dataset_name="real_dataset",
        dataset_version="1.0",
        start_date=Arrow(2025, 6, 1),
        end_date=Arrow(2025, 6, 1),
        filter_bounds=(Lon(70.0), Lat(8.0), Lon(90.0), Lat(37.0)),
        variable_mapping={
            "var1": "var1_target",
            "var2": "var2_target",
        },
    )

    # Create the pipeline
    pipeline = NedExportPipeline(
        grid=mock_grid,
        archive_storage=mock_archive_storage,
        dataset_descriptor=real_dataset_descriptor,
        dataset_retriever=mock_dataset_retriever,
        dataset_reader=mock_dataset_reader,
        result_subpath="mock/subpath",
    )

    # Run the upload method
    pipeline.upload()

    # Assertions
    mock_archive_storage.write_to_destination.assert_called_once()
    written_table = mock_archive_storage.write_to_destination.call_args[1]["table"]

    manually_computed_expected_values_var1 = [
        1.75,
        3.25,
        4.75,
        6.25,
    ]

    manually_computed_expected_values_var2 = [
        17.5,
        32.5,
        47.5,
        62.5,
    ]

    # Check that the regridded data matches the expected values
    expected_data = pl.DataFrame(
        {
            "grid_id": ["01", "02", "03", "04"],
            "date": ["2025-06-01"] * 4,
            "var1_target": manually_computed_expected_values_var1,
            "var2_target": manually_computed_expected_values_var2,
        }
    )

    assert_frame_equal(
        written_table,
        expected_data,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        rtol=0,
        atol=0.01,
    )


def test__NedExportPipeline__missing_dates__fills_in_missing_rows():
    # Create a mock grid (4x4 grid)
    mock_grid = Grid(
        DataFrame(
            {
                "grid_id": ["01", "02", "03", "04"],
                "lon": [0.5, 1.5, 2.5, 3.5],
                "lat": [2.5, 3.5, 4.5, 5.5],
            }
        )
    )

    # Create mock data with missing dates
    mock_data = NedDayData(
        dataset=xr.Dataset(
            {
                "mock_var": (
                    ["lat", "lon"],
                    [
                        [1.0, 2.0, 3.0],
                        [3.0, 4.0, 5.0],
                        [5.0, 6.0, 7.0],
                    ],
                )
            },
            coords={"lat": [2.0, 4.0, 6.0], "lon": [0.0, 2.0, 4.0]},
        ),
        date="2025-06-01",
    )

    # Mock the dataset retriever
    mock_dataset_retriever = MagicMock()
    mock_dataset_retriever.stream_files.return_value = [MagicMock()]

    # Mock the dataset reader
    mock_dataset_reader = MagicMock()
    mock_dataset_reader.extract_data.return_value = mock_data

    # Mock the archive storage
    mock_archive_storage = MagicMock()
    mock_archive_storage.write_to_destination.side_effect = lambda table, result_subpath: table

    # Use a real dataset descriptor
    real_dataset_descriptor = NedDatasetDescriptor(
        dataset_name="real_dataset",
        dataset_version="1.0",
        start_date=Arrow(2025, 6, 1),
        end_date=Arrow(2025, 6, 3),
        filter_bounds=(Lon(70.0), Lat(8.0), Lon(90.0), Lat(37.0)),
        variable_mapping={"mock_var": "mock_var_target"},
    )

    # Create the pipeline
    pipeline = NedExportPipeline(
        grid=mock_grid,
        archive_storage=mock_archive_storage,
        dataset_descriptor=real_dataset_descriptor,
        dataset_retriever=mock_dataset_retriever,
        dataset_reader=mock_dataset_reader,
        result_subpath="mock/subpath",
    )

    # Run the upload method
    pipeline.upload()

    # Assertions
    mock_archive_storage.write_to_destination.assert_called_once()
    written_table = mock_archive_storage.write_to_destination.call_args[1]["table"]

    manually_computed_expected_values = [
        1.75,
        None,
        None,
        6.25,
        None,
        None,
        4.75,
        None,
        None,
        3.25,
        None,
        None,
    ]

    # Check that the regridded data matches the expected values, including missing rows
    expected_data = pl.DataFrame(
        {
            "grid_id": ["01", "02", "03", "04"] * 3,
            "date": ["2025-06-01", "2025-06-02", "2025-06-03"] * 4,
            "mock_var_target": manually_computed_expected_values,
        }
    )

    assert_frame_equal(
        written_table,
        expected_data,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        rtol=0,
        atol=0.01,
    )
