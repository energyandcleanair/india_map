from unittest.mock import MagicMock, call, patch

import pyarrow as pa
import pytest

from pm25ml.collectors.export_pipeline import GeeExportPipeline
from pm25ml.collectors.feature_planner import FeaturePlan

CURRENT_TIME = "20250529_123456"
FIXED_UUID = "12345678-1234-5678-1234-567812345678"


@pytest.fixture(autouse=True)
def mock_now():
    with patch("pm25ml.collectors.export_pipeline.now") as mock_now:
        mock_now.return_value.strftime.return_value = CURRENT_TIME
        yield mock_now


@pytest.fixture(autouse=True)
def mock_uuid():
    with patch("pm25ml.collectors.export_pipeline.uuid.uuid4") as mock_uuid:
        mock_uuid.return_value = FIXED_UUID
        yield mock_uuid


@pytest.fixture(autouse=True)
def mock_sleep():
    with patch("pm25ml.collectors.export_pipeline.sleep") as mock_sleep:
        yield mock_sleep


@pytest.fixture
def example_feature_plan():
    planned_collection = MagicMock()
    return FeaturePlan(
        feature_type="mock_type",
        column_mappings={"col1": "mapped_col1", "col2": "mapped_col2"},
        planned_collection=planned_collection,
    )


@pytest.fixture
def mock_storage_valid_table():
    data = {
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
        "extra_col": [7, 8, 9],
    }
    table = pa.Table.from_pydict(data)

    storage = MagicMock()
    storage.intermediate_bucket = "mock_bucket"
    storage.get_intermediate_by_id.return_value = table
    return storage


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
    with patch("pm25ml.collectors.export_pipeline.Export") as MockExport:
        MockExport.table.toCloudStorage.return_value = mock_task
        yield MockExport


@pytest.fixture
def mock_storage_missing_columns():
    data = {
        "col1": [1, 2, 3],
        "extra_col": [7, 8, 9],
    }
    table = pa.Table.from_pydict(data)
    storage = MagicMock()
    storage.get_intermediate_by_id.return_value = table
    return storage


@pytest.fixture
def mock_storage_all_null_values():
    data = {
        "col1": [1, 2, 3],
        "col2": [None, None, None],
        "extra_col": [7, 8, 9],
    }
    table = pa.Table.from_pydict(data)
    storage = MagicMock()
    storage.get_intermediate_by_id.return_value = table
    return storage


def test_GeeExportPipeline_upload_taskExecutionAndStorageHandling(
    mock_storage_valid_table, example_feature_plan, mock_successful_export, mock_task,
) -> None:
    pipeline = GeeExportPipeline(
        gee_export_pipeline_storage=mock_storage_valid_table,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    pipeline.upload()

    # Check that the task was defined correctly
    mock_successful_export.table.toCloudStorage.assert_called_once_with(
        description=f"mock_type_{FIXED_UUID}_{CURRENT_TIME}",
        collection=example_feature_plan.planned_collection,
        bucket="mock_bucket",
        fileNamePrefix=FIXED_UUID,
        fileFormat="CSV",
        selectors=["col1", "col2"],
    )

    # Check that the task was started and completed
    mock_task.start.assert_called_once()
    mock_task.active.assert_called()
    mock_task.status.assert_called()

    # Check that intermediate storage methods were called
    mock_storage_valid_table.get_intermediate_by_id.assert_called_once_with(FIXED_UUID)
    mock_storage_valid_table.write_to_destination.assert_called_once()
    mock_storage_valid_table.delete_intermediate_by_id.assert_called_once_with(FIXED_UUID)


def test_GeeExportPipeline_upload_taskFailure(
    mock_storage_valid_table, example_feature_plan, mock_task,
) -> None:
    # Simulate a task ending with a FAILURE status
    mock_task.status.return_value = {
        "state": "FAILED",
        "error_message": "Task failed due to an unknown error.",
    }

    pipeline = GeeExportPipeline(
        gee_export_pipeline_storage=mock_storage_valid_table,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    with pytest.raises(RuntimeError, match="Task failed due to an unknown error."):
        pipeline.upload()


def test_GeeExportPipeline_process_tableProcessingLogic(
    mock_storage_valid_table, example_feature_plan,
) -> None:
    pipeline = GeeExportPipeline(
        gee_export_pipeline_storage=mock_storage_valid_table,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    # Call the public upload method
    pipeline.upload()

    # Validate that the processed table has the expected structure
    processed_table = mock_storage_valid_table.write_to_destination.call_args[0][0]

    # Check that extra columns were dropped
    assert "extra_col" not in processed_table.column_names

    # Check that the table was sorted
    expected_data = {
        "mapped_col1": [1, 2, 3],
        "mapped_col2": [4, 5, 6],
    }
    expected_table = pa.Table.from_pydict(expected_data)
    assert processed_table.equals(expected_table)


def test_GeeExportPipeline_upload_exponentialBackoff(
    mock_storage_valid_table, example_feature_plan, mock_sleep,
) -> None:
    pipeline = GeeExportPipeline(
        gee_export_pipeline_storage=mock_storage_valid_table,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    with patch("pm25ml.collectors.export_pipeline.Export") as MockExport:
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
    mock_storage_missing_columns, example_feature_plan,
) -> None:
    pipeline = GeeExportPipeline(
        gee_export_pipeline_storage=mock_storage_missing_columns,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    with pytest.raises(ValueError, match="Table is missing expected columns: col2"):
        pipeline.upload()


def test_GeeExportPipeline_upload_allNullColumns(
    mock_storage_all_null_values, example_feature_plan,
) -> None:
    pipeline = GeeExportPipeline(
        gee_export_pipeline_storage=mock_storage_all_null_values,
        plan=example_feature_plan,
        result_subpath="mock/result/path",
    )

    with pytest.raises(ValueError, match="Table has columns with all null values: col2"):
        pipeline.upload()
