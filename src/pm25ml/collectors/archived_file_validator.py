"""Validator for the metadata of export results."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyarrow import Schema, float32, float64, int64, large_string

from pm25ml.logging import logger

if TYPE_CHECKING:
    from pm25ml.collectors.archive_storage import IngestArchiveStorage
    from pm25ml.collectors.collector import UploadResult
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

    def validate_all_results(self, results: list[UploadResult]) -> None:
        """
        Validate the schema of all results against their expected schemas.

        :param results: A list of PipelineConfig objects containing the expected results.
        :raises ValueError: If any result's schema does not match the expected schema.
        """
        import concurrent.futures

        @dataclass
        class _ValidationResult:
            result: UploadResult
            exception: Exception | None = None
            traceback: str | None = None

            def __str__(self) -> str:
                metadata = self.result.pipeline_config
                return f"{metadata.result_subpath}: {self.traceback}"

        def validate_pipeline(result: UploadResult) -> _ValidationResult:
            try:
                self.validate_result_schema(result)
            except Exception as e:  # noqa: BLE001
                return _ValidationResult(
                    result=result,
                    exception=e,
                    traceback=traceback.format_exc(),
                )
            else:
                return _ValidationResult(result=result)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            validations = executor.map(validate_pipeline, results)
            validation_results = list(validations)
            errored_results = [
                result for result in validation_results if result.exception is not None
            ]

        if errored_results:
            error_msgs = "\n".join(str(errored_result) for errored_result in errored_results)
            msg = f"Validation failed for {len(errored_results)} pipeline(s):\n{error_msgs}"
            raise ArchivedFileValidatorError(
                msg,
            )

    def validate_result_schema(self, expected_result: UploadResult) -> None:
        """
        Validate the schema of the result against the expected schema.

        :param result: The ExportResult object containing the result subpath and expected columns.
        :raises ValueError: If the actual schema does not match the expected schema.
        """
        config = expected_result.pipeline_config
        if not expected_result.completeness.data_available and config.allows_missing_data:
            logger.info(
                f"Skipping validation for {config.result_subpath} as it "
                "is empty and allows missing data.",
            )
            return

        self._validate_expected_against_actual(config)

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
        try:
            file_metadata = self.archive_storage.read_dataframe_metadata(
                result_subpath=expected_result.result_subpath,
            )
        except FileNotFoundError as exc:
            msg = (
                f"Result {expected_result.result_subpath} does not exist in archive storage. "
                "It needs to be uploaded."
            )
            raise ArchivedFileValidatorError(msg) from exc

        rows = file_metadata.num_rows
        self._check_count_rows(expected_result, rows)

        actual_schema = file_metadata.schema.to_arrow_schema()
        as_str = str(actual_schema).replace("\n", " | ")
        logger.debug(
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
