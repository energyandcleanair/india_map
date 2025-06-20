from unittest.mock import MagicMock, call, patch

import pytest
from polars import DataFrame

from pm25ml.collectors.gee.intermediate_storage import GeeIntermediateStorage
from pm25ml.collectors.pipeline_storage import IngestArchiveStorage

from .feature_planner import FeaturePlan
from .gee_export_pipeline import GeeExportPipeline

FIXED_NANO_ID = "12345678-1234-5678-1234-567812345678"


@pytest.fixture(autouse=True)
def mock_nanoid():
    with patch("pm25ml.collectors.gee.gee_export_pipeline.generate") as mock_nanoid:
        mock_nanoid.return_value = FIXED_NANO_ID
        yield mock_nanoid


@pytest.fixture(autouse=True)
def mock_sleep():
    with patch("pm25ml.collectors.gee.gee_export_pipeline.sleep") as mock_sleep:
        yield mock_sleep


@pytest.fixture
def example_feature_plan():
    planned_collection = MagicMock()
    return FeaturePlan(
        feature_name="mock_type",
        column_mappings={"col1": "mapped_col1", "col2": "mapped_col2"},
        planned_collection=planned_collection,
    )


@pytest.fixture
def mock_task():
    task = MagicMock()
    task.active.side_effect = [
        True,
        False,
    ]  # Simulate task becoming inactive after one iteration
    task.status.return_value = {"state": "COMPLETED"}
    return task


# This is autoused by default but can be overridden in tests if needed
@pytest.fixture(autouse=True)
def mock_successful_export(mock_task):
    with patch("pm25ml.collectors.gee.gee_export_pipeline.Export") as MockExport:
        MockExport.table.toCloudStorage.return_value = mock_task
        yield MockExport


@pytest.fixture
def mock_intermediate_storage_valid_table():
    data = {
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
        "extra_col": [7, 8, 9],
    }
    table = DataFrame(data)

    storage = MagicMock(spec=GeeIntermediateStorage)
    storage.bucket = "mock_bucket"
    storage.get_intermediate_by_id.return_value = table
    return storage


@pytest.fixture
def mock_intermediate_storage_missing_columns():
    data = {
        "col1": [1, 2, 3],
        "extra_col": [7, 8, 9],
    }
    table = DataFrame(data)
    storage = MagicMock(spec=GeeIntermediateStorage)
    storage.bucket = "mock_bucket"
    storage.get_intermediate_by_id.return_value = table
    return storage


@pytest.fixture
def mock_intermediate_storage_all_null_values():
    data = {
        "col1": [1, 2, 3],
        "col2": [None, None, None],
        "extra_col": [7, 8, 9],
    }
    table = DataFrame(data)
    storage = MagicMock(spec=GeeIntermediateStorage)
    storage.bucket = "mock_bucket"
    storage.get_intermediate_by_id.return_value = table
    return storage


@pytest.fixture
def mock_archive_storage():
    storage = MagicMock(spec=IngestArchiveStorage)
    storage.destination_bucket = "mock_destination_bucket"
    return storage


@pytest.fixture
def example_plan_with_date_and_Grid():
    planned_collection = MagicMock()

    return FeaturePlan(
        feature_name="mock_name",
        column_mappings={
            "date": "date",
            "grid_id": "grid_id",
            "col1": "mapped_col1",
            "col2": "mapped_col2",
        },
        planned_collection=planned_collection,
    )


@pytest.fixture
def mock_intermediate_storage_out_of_order():
    data = {
        "col1": [3, 1, 2, 4],
        "col2": [6, 4, 5, 4],
        "date": ["2025-06-03", "2025-06-01", "2025-06-02", "2025-06-01"],
        "grid_id": ["grid_1", "grid_1", "grid_2", "grid_3"],
    }
    table = DataFrame(data)

    storage = MagicMock(spec=GeeIntermediateStorage)
    storage.bucket = "mock_bucket"
    storage.get_intermediate_by_id.return_value = table
    return storage


def test_GeeExportPipeline_upload_taskExecutionAndStorageHandling(
    mock_intermediate_storage_valid_table,
    example_feature_plan,
    mock_successful_export,
    mock_task,
    mock_archive_storage,
) -> None:
    pipeline = GeeExportPipeline(
        archive_storage=mock_archive_storage,
        intermediate_storage=mock_intermediate_storage_valid_table,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    pipeline.upload()

    file_name = f"{FIXED_NANO_ID}__mock_type"

    # Check that the task was defined correctly
    mock_successful_export.table.toCloudStorage.assert_called_once_with(
        description=file_name,
        collection=example_feature_plan.planned_collection,
        bucket="mock_bucket",
        fileNamePrefix=file_name,
        fileFormat="CSV",
        selectors=["col1", "col2"],
    )

    # Check that the task was started and completed
    mock_task.start.assert_called_once()
    mock_task.active.assert_called()
    mock_task.status.assert_called()

    # Check that intermediate storage methods were called
    mock_intermediate_storage_valid_table.get_intermediate_by_id.assert_called_once_with(file_name)
    mock_intermediate_storage_valid_table.delete_intermediate_by_id.assert_called_once_with(
        file_name,
    )
    mock_archive_storage.write_to_destination.assert_called_once()


def test_GeeExportPipeline_upload_taskFailure(
    mock_intermediate_storage_valid_table,
    example_feature_plan,
    mock_task,
    mock_archive_storage,
) -> None:
    # Simulate a task ending with a FAILURE status
    mock_task.status.return_value = {
        "state": "FAILED",
        "error_message": "Task failed due to an unknown error.",
    }

    pipeline = GeeExportPipeline(
        archive_storage=mock_archive_storage,
        intermediate_storage=mock_intermediate_storage_valid_table,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    with pytest.raises(RuntimeError, match="Task failed due to an unknown error."):
        pipeline.upload()


def test_GeeExportPipeline_process_tableProcessingLogic(
    mock_intermediate_storage_valid_table,
    example_feature_plan,
    mock_archive_storage,
) -> None:
    pipeline = GeeExportPipeline(
        archive_storage=mock_archive_storage,
        intermediate_storage=mock_intermediate_storage_valid_table,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    # Call the public upload method
    pipeline.upload()

    # Validate that the processed table has the expected structure
    processed_table = mock_archive_storage.write_to_destination.call_args[0][0]

    # Check that extra columns were dropped
    assert "extra_col" not in processed_table.columns

    # Check that the table was sorted
    expected_data = {
        "mapped_col1": [1, 2, 3],
        "mapped_col2": [4, 5, 6],
    }
    expected_table = DataFrame(expected_data)
    assert processed_table.equals(expected_table)


def test_GeeExportPipeline_upload_exponentialBackoff(
    mock_intermediate_storage_valid_table,
    example_feature_plan,
    mock_sleep,
    mock_archive_storage,
) -> None:
    pipeline = GeeExportPipeline(
        archive_storage=mock_archive_storage,
        intermediate_storage=mock_intermediate_storage_valid_table,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    with patch("pm25ml.collectors.gee.gee_export_pipeline.Export") as MockExport:
        mock_task = MagicMock()
        MockExport.table.toCloudStorage.return_value = mock_task

        # Simulate task being active for a few iterations before completing
        mock_task.active.side_effect = [True, True, False]
        mock_task.status.return_value = {"state": "COMPLETED"}

        pipeline.upload()

        # Check that sleep was called with exponential backoff values
        mock_sleep.assert_has_calls(
            [
                call(1),
                call(1.5),
            ],
        )


def test_GeeExportPipeline_upload_missingColumns(
    mock_intermediate_storage_missing_columns,
    example_feature_plan,
    mock_archive_storage,
) -> None:
    pipeline = GeeExportPipeline(
        archive_storage=mock_archive_storage,
        intermediate_storage=mock_intermediate_storage_missing_columns,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    with pytest.raises(ValueError, match="Table is missing expected columns: col2"):
        pipeline.upload()


def test_GeeExportPipeline_upload_allNullColumns(
    mock_intermediate_storage_all_null_values,
    example_feature_plan,
    mock_archive_storage,
) -> None:
    pipeline = GeeExportPipeline(
        archive_storage=mock_archive_storage,
        intermediate_storage=mock_intermediate_storage_all_null_values,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    with pytest.raises(ValueError, match="Table has columns with all null values: col2"):
        pipeline.upload()


def test_GeeExportPipeline_process_tableSortingByDateAndGridId_outOfOrder(
    mock_intermediate_storage_out_of_order,
    example_plan_with_date_and_Grid,
    mock_archive_storage,
) -> None:
    pipeline = GeeExportPipeline(
        archive_storage=mock_archive_storage,
        intermediate_storage=mock_intermediate_storage_out_of_order,
        plan=example_plan_with_date_and_Grid,
        result_subpath="mock/result/path",
    )

    # Call the public upload method
    pipeline.upload()

    # Validate that the processed table is sorted by date and then grid_id
    processed_table: DataFrame = mock_archive_storage.write_to_destination.call_args[0][0]

    # Check that the table was sorted by date and then grid_id
    sorted_table = processed_table.sort(["date", "grid_id"])
    assert processed_table.equals(sorted_table)
