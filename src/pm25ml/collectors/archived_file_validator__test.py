"""Unit tests for ArchivedFileValidator."""

import pytest
from pyarrow import schema, field, int64, float64, large_string
from unittest.mock import Mock

from pm25ml.collectors.archived_file_validator import (
    ArchivedFileValidator,
    ArchivedFileValidatorError,
)
from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.collector import DataCompleteness, UploadResult
from pm25ml.collectors.export_pipeline import (
    MissingDataHeuristic,
    PipelineConfig,
    PipelineConsumerBehaviour,
)


def test__validate_result_schema__valid_schema__no_error():
    """Test validate_result_schema with valid schema and row count."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    file_metadata_mock = Mock()
    file_metadata_mock.num_rows = 100
    file_metadata_mock.schema.to_arrow_schema.return_value = schema(
        [
            field("grid_id", int64()),
            field("date", large_string()),
            field("value_column", float64()),
        ]
    )
    archive_storage_mock.read_dataframe_metadata.return_value = file_metadata_mock

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = UploadResult(
        PipelineConfig(
            result_subpath="path/to/result",
            expected_rows=100,
            id_columns={"grid_id", "date"},
            value_columns={"value_column"},
        ),
        completeness=DataCompleteness.COMPLETE,
        error=None,
    )

    validator.validate_result_schema(expected_result)


def test__validate_result_schema__invalid_row_count__raises_error():
    """Test validate_result_schema with invalid row count."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    file_metadata_mock = Mock()
    file_metadata_mock.num_rows = 50
    archive_storage_mock.read_dataframe_metadata.return_value = file_metadata_mock

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = UploadResult(
        PipelineConfig(
            result_subpath="path/to/result",
            expected_rows=100,
            id_columns={"grid_id", "date"},
            value_columns={"value_column"},
        ),
        completeness=DataCompleteness.COMPLETE,
        error=None,
    )

    with pytest.raises(
        ArchivedFileValidatorError, match="Expected 100 rows in path/to/result, but found 50 rows."
    ):
        validator.validate_result_schema(expected_result)


def test__validate_result_schema__missing_column__raises_error():
    """Test validate_result_schema with missing column."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    file_metadata_mock = Mock()
    file_metadata_mock.num_rows = 100
    file_metadata_mock.schema.to_arrow_schema.return_value = schema(
        [
            field("grid_id", int64()),
            field("value_column", float64()),
        ]
    )
    archive_storage_mock.read_dataframe_metadata.return_value = file_metadata_mock

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = UploadResult(
        PipelineConfig(
            result_subpath="path/to/result",
            expected_rows=100,
            id_columns={"grid_id", "date"},
            value_columns={"value_column"},
        ),
        completeness=DataCompleteness.COMPLETE,
        error=None,
    )

    with pytest.raises(
        ArchivedFileValidatorError,
        match="Expected 'date' column in path/to/result, but it is missing.",
    ):
        validator.validate_result_schema(expected_result)


def test__validate_result_schema__invalid_grid_column_type__raises_error():
    """Test validate_result_schema with invalid column type."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    file_metadata_mock = Mock()
    file_metadata_mock.num_rows = 100
    file_metadata_mock.schema.to_arrow_schema.return_value = schema(
        [
            field("grid_id", float64()),
            field("date", large_string()),
            field("value_column", float64()),
        ]
    )
    archive_storage_mock.read_dataframe_metadata.return_value = file_metadata_mock

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = UploadResult(
        PipelineConfig(
            result_subpath="path/to/result",
            expected_rows=100,
            id_columns={"grid_id", "date"},
            value_columns={"value_column"},
        ),
        completeness=DataCompleteness.COMPLETE,
        error=None,
    )

    with pytest.raises(
        ArchivedFileValidatorError,
        match="Expected 'grid_id' column to be of type int64 in path/to/result, but found (double|float).*.",
    ):
        validator.validate_result_schema(expected_result)


def test__validate_result_schema__file_missing__raises_error():
    """Test validate_result_schema when file is missing."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    archive_storage_mock.read_dataframe_metadata.side_effect = FileNotFoundError("File not found")

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = UploadResult(
        PipelineConfig(
            result_subpath="path/to/result",
            expected_rows=100,
            id_columns={"grid_id", "date"},
            value_columns={"value_column"},
        ),
        completeness=DataCompleteness.COMPLETE,
        error=None,
    )

    with pytest.raises(
        ArchivedFileValidatorError,
        match="Result path/to/result does not exist in archive storage. It needs to be uploaded.",
    ):
        validator.validate_result_schema(expected_result)


def test__validate_result_schema__missing_but_allowed__no_error():
    """Test validate_result_schema with allowed missing columns."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    archive_storage_mock.read_dataframe_metadata.side_effect = FileNotFoundError("File not found")

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = UploadResult(
        PipelineConfig(
            result_subpath="path/to/result",
            expected_rows=100,
            id_columns={"grid_id", "date"},
            value_columns=set(),
            consumer_behaviour=PipelineConsumerBehaviour(
                missing_data_heuristic=MissingDataHeuristic.COPY_LATEST_AVAILABLE_BEFORE
            ),
        ),
        completeness=DataCompleteness.EMPTY,
        error=None,
    )

    validator.validate_result_schema(expected_result)


def test__validate_result_schema__invalid_date_column_type__raises_error():
    """Test validate_result_schema with invalid date column type."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    file_metadata_mock = Mock()
    file_metadata_mock.num_rows = 100
    file_metadata_mock.schema.to_arrow_schema.return_value = schema(
        [
            field("grid_id", int64()),
            field("date", int64()),  # Incorrect type
            field("value_column", float64()),
        ]
    )
    archive_storage_mock.read_dataframe_metadata.return_value = file_metadata_mock

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = UploadResult(
        PipelineConfig(
            result_subpath="path/to/result",
            expected_rows=100,
            id_columns={"grid_id", "date"},
            value_columns={"value_column"},
        ),
        completeness=DataCompleteness.COMPLETE,
        error=None,
    )

    with pytest.raises(
        ArchivedFileValidatorError,
        match="Expected 'date' column to be of type string in path/to/result, but found int64.",
    ):
        validator.validate_result_schema(expected_result)


def test__validate_result_schema__invalid_value_column__raises_error():
    """Test validate_result_schema with invalid value column."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    file_metadata_mock = Mock()
    file_metadata_mock.num_rows = 100
    file_metadata_mock.schema.to_arrow_schema.return_value = schema(
        [
            field("grid_id", int64()),
            field("date", large_string()),
            field("value_column", int64()),  # Incorrect type
        ]
    )
    archive_storage_mock.read_dataframe_metadata.return_value = file_metadata_mock

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = UploadResult(
        PipelineConfig(
            result_subpath="path/to/result",
            expected_rows=100,
            id_columns={"grid_id", "date"},
            value_columns={"value_column"},
        ),
        completeness=DataCompleteness.COMPLETE,
        error=None,
    )

    with pytest.raises(
        ArchivedFileValidatorError,
        match="Expected 'value_column' column to be of type .* in path/to/result, but found int64.",
    ):
        validator.validate_result_schema(expected_result)


def test__needs_upload__missing_in_archive__returns_true():
    """Test needs_upload when the dataset is missing in archive storage."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    archive_storage_mock.does_dataset_exist.return_value = False

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = PipelineConfig(
        result_subpath="path/to/result",
        expected_rows=100,
        id_columns={"grid_id", "date"},
        value_columns={"value_column"},
    )

    assert validator.needs_upload(expected_result) is True


def test__needs_upload__validation_fails__returns_true():
    """Test needs_upload when validation fails."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    archive_storage_mock.does_dataset_exist.return_value = True
    file_metadata_mock = Mock()
    file_metadata_mock.num_rows = 100
    file_metadata_mock.schema.to_arrow_schema.return_value = schema(
        [
            field("grid_id", float64()),  # Invalid type
            field("date", large_string()),
            field("value_column", float64()),
        ]
    )
    archive_storage_mock.read_dataframe_metadata.return_value = file_metadata_mock

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = PipelineConfig(
        result_subpath="path/to/result",
        expected_rows=100,
        id_columns={"grid_id", "date"},
        value_columns={"value_column"},
    )

    assert validator.needs_upload(expected_result) is True


def test__needs_upload__validation_passes__returns_false():
    """Test needs_upload when validation passes."""
    archive_storage_mock = Mock(spec=IngestArchiveStorage)
    archive_storage_mock.does_dataset_exist.return_value = True
    file_metadata_mock = Mock()
    file_metadata_mock.num_rows = 100
    file_metadata_mock.schema.to_arrow_schema.return_value = schema(
        [
            field("grid_id", int64()),
            field("date", large_string()),
            field("value_column", float64()),
        ]
    )
    archive_storage_mock.read_dataframe_metadata.return_value = file_metadata_mock

    validator = ArchivedFileValidator(archive_storage=archive_storage_mock)

    expected_result = PipelineConfig(
        result_subpath="path/to/result",
        expected_rows=100,
        id_columns={"grid_id", "date"},
        value_columns={"value_column"},
    )

    assert validator.needs_upload(expected_result) is False
