"""Validator for the metadata of export results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyarrow import Schema, float32, float64, int64, large_string

from pm25ml.logging import logger

if TYPE_CHECKING:
    from pm25ml.collectors.archive_storage import IngestArchiveStorage
    from pm25ml.collectors.export_pipeline import PipelineConfig


class ArchivedFileValidatorError(Exception):
    """Base exception for errors in the ArchivedFileValidator."""


class SchemaMismatchError(ArchivedFileValidatorError):
    """Exception raised when the actual schema does not match the expected schema."""


class MissingValueColumnError(SchemaMismatchError):
    """Exception raised when a value column is missing in the result schema."""


class MissingIdColumnError(SchemaMismatchError):
    """Exception raised when an ID column is missing in the result schema."""


class IncorrectColumnTypeError(SchemaMismatchError):
    """Exception raised when a column type does not match the expected type."""


class IncorrectNumberOfRowsError(ArchivedFileValidatorError):
    """Exception raised when the number of rows in the result does not match the expected count."""


class ArchivedFileValidator:
    """Validator for the metadata of export results."""

    def __init__(self, archive_storage: IngestArchiveStorage) -> None:
        """
        Initialize the MetadataValidator with the archive storage.

        :param archive_storage: The storage where the results are archived.
        """
        self.archive_storage = archive_storage

    def validate_all_results(self, pipelines: list[PipelineConfig]) -> None:
        """
        Validate the schema of all results against their expected schemas.

        :param pipelines: A list of PipelineConfig objects containing the expected results.
        :raises ValueError: If any result's schema does not match the expected schema.
        """
        import concurrent.futures

        errors: list[tuple[PipelineConfig, Exception]] = []

        def validate_pipeline(pipeline: PipelineConfig) -> tuple[PipelineConfig, Exception | None]:
            try:
                self.validate_result_schema(pipeline)
            except Exception as e:  # noqa: BLE001
                return (pipeline, e)
            else:
                return (pipeline, None)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(validate_pipeline, pipeline) for pipeline in pipelines]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result[1] is not None:
                    error_result = (result[0], result[1])
                    errors.append(error_result)

        if errors:
            error_msgs = "\n".join(f"{pipeline.result_subpath}: {exc}" for pipeline, exc in errors)
            msg = f"Validation failed for {len(errors)} pipeline(s):\n{error_msgs}"
            raise ArchivedFileValidatorError(
                msg,
            )

    def validate_result_schema(self, expected_result: PipelineConfig) -> None:
        """
        Validate the schema of the result against the expected schema.

        :param result: The ExportResult object containing the result subpath and expected columns.
        :raises ValueError: If the actual schema does not match the expected schema.
        """
        self._validate_expected_against_actual(expected_result)

    def needs_upload(self, expected_result: PipelineConfig) -> bool:
        """
        Check if the result needs to be uploaded.

        It needs to be uploaded if:
        1. It doesn't already exist in the archive storage.
        2. It fails validation against the expected schema.

        :param expected_result: The expected result configuration.
        :return: True if the result needs to be uploaded, False otherwise.
        """
        if not self.archive_storage.does_dataset_exist(expected_result.result_subpath):
            logger.info(
                f"Result {expected_result.result_subpath} needs upload: "
                f"does not exist in archive storage.",
            )
            return True
        try:
            self._validate_expected_against_actual(expected_result)
        except ArchivedFileValidatorError:
            logger.info(
                f"Result {expected_result.result_subpath} needs upload: validation failed.",
                exc_info=True,
            )
            return True
        logger.debug(f"Result {expected_result.result_subpath} does not need upload")
        return False

    def _validate_expected_against_actual(self, expected_result: PipelineConfig) -> None:
        file_metadata = self.archive_storage.read_dataframe_metadata(
            result_subpath=expected_result.result_subpath,
        )

        rows = file_metadata.num_rows
        self._check_count_rows(expected_result, rows)

        actual_schema = file_metadata.schema.to_arrow_schema()
        as_str = str(actual_schema).replace("\n", " | ")
        logger.info(
            f"Result {expected_result.result_subpath} has {rows} rows and schema: [{as_str}]",
        )
        self._check_schema(expected_result, actual_schema)

    def _check_count_rows(self, result: PipelineConfig, rows: int) -> None:
        if rows != result.expected_rows:
            msg = (
                f"Expected {result.expected_rows} rows in {result.result_subpath}, "
                f"but found {rows} rows."
            )
            raise IncorrectNumberOfRowsError(msg)

    def _check_schema(self, result: PipelineConfig, actual_schema: Schema) -> None:
        if "grid_id" in result.id_columns:
            try:
                actual_grid_id_column = actual_schema.field("grid_id")
            except KeyError as exc:
                msg = f"Expected 'grid_id' column in {result.result_subpath}, but it is missing."
                raise MissingIdColumnError(msg) from exc

            if actual_grid_id_column.type != int64():
                msg = (
                    f"Expected 'grid_id' column to be of type int64 in {result.result_subpath}, "
                    f"but found {actual_grid_id_column.type}."
                )
                raise IncorrectColumnTypeError(msg)

        if "date" in result.id_columns:
            try:
                actual_date_column = actual_schema.field("date")
            except KeyError as exc:
                msg = f"Expected 'date' column in {result.result_subpath}, but it is missing."
                raise MissingIdColumnError(msg) from exc

            if actual_date_column.type != large_string():
                msg = (
                    f"Expected 'date' column to be of type string in {result.result_subpath}, "
                    f"but found {actual_date_column.type} (expected large_string)."
                )
                raise IncorrectColumnTypeError(msg)

        for value_column in result.value_columns:
            try:
                actual_value_column = actual_schema.field(value_column)
            except KeyError as exc:
                msg = (
                    f"Expected value column '{value_column}' in {result.result_subpath}, "
                    "but it is missing."
                )
                raise MissingValueColumnError(msg) from exc

            allowed_value_types = {float64(), float32()}
            if actual_value_column.type not in allowed_value_types:
                msg = (
                    f"Expected '{value_column}' column to be of type of {allowed_value_types} in "
                    f"{result.result_subpath}, but found {actual_value_column.type}."
                )
                raise IncorrectColumnTypeError(msg)
