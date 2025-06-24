"""Validator for the metadata of export results."""

from pyarrow import Schema, float64, int64, large_string

from pm25ml.collectors.export_pipeline import PipelineConfig
from pm25ml.collectors.pipeline_storage import IngestArchiveStorage
from pm25ml.logging import logger


class ArchivedFileValidator:
    """Validator for the metadata of export results."""

    def __init__(self, archive_storage: IngestArchiveStorage) -> None:
        """
        Initialize the MetadataValidator with the archive storage.

        :param archive_storage: The storage where the results are archived.
        """
        self.archive_storage = archive_storage

    def validate_result_schema(self, expected_result: PipelineConfig) -> None:
        """
        Validate the schema of the result against the expected schema.

        :param result: The ExportResult object containing the result subpath and expected columns.
        :raises ValueError: If the actual schema does not match the expected schema.
        """
        self._validate_expected_against_actual(expected_result)

    def _validate_expected_against_actual(self, expected_result: PipelineConfig) -> None:
        file_metadata = self.archive_storage.read_dataframe_metadata(
            result_subpath=expected_result.result_subpath,
        )

        rows = file_metadata.num_rows
        self._check_count_rows(expected_result, rows)

        actual_schema = file_metadata.schema.to_arrow_schema()
        logger.info(
            f"Result {expected_result.result_subpath} has {rows} rows and schema: {actual_schema}",
        )
        self._check_schema(expected_result, actual_schema)

    def _check_count_rows(self, result: PipelineConfig, rows: int) -> None:
        if rows != result.expected_rows:
            msg = (
                f"Expected {result.expected_rows} rows in {result.result_subpath}, "
                f"but found {rows} rows."
            )
            raise ValueError(
                msg,
            )

    def _check_schema(self, result: PipelineConfig, actual_schema: Schema) -> None:
        if "grid_id" in result.id_columns:
            try:
                actual_grid_id_column = actual_schema.field("grid_id")
            except KeyError as exc:
                msg = f"Expected 'grid_id' column in {result.result_subpath}, but it is missing."
                raise ValueError(msg) from exc

            if actual_grid_id_column.type != int64():
                msg = (
                    f"Expected 'grid_id' column to be of type int64 in {result.result_subpath}, "
                    f"but found {actual_grid_id_column.type}."
                )
                raise ValueError(msg)

        if "date" in result.id_columns:
            try:
                actual_date_column = actual_schema.field("date")
            except KeyError as exc:
                msg = f"Expected 'date' column in {result.result_subpath}, but it is missing."
                raise ValueError(msg) from exc

            if actual_date_column.type != large_string():
                msg = (
                    f"Expected 'date' column to be of type string in {result.result_subpath}, "
                    f"but found {actual_date_column.type} (expected large_string)."
                )
                raise ValueError(msg)

        for value_column in result.value_columns:
            try:
                actual_value_column = actual_schema.field(value_column)
            except KeyError as exc:
                msg = (
                    f"Expected value column '{value_column}' in {result.result_subpath}, "
                    "but it is missing."
                )
                raise ValueError(msg) from exc

            if actual_value_column.type != float64():
                msg = (
                    f"Expected '{value_column}' column to be of type float64 in "
                    f"{result.result_subpath}, but found {actual_value_column.type}."
                )
                raise ValueError(msg)
