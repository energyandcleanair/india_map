"""Handles the collection of raw data from various sources to the archive storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING

from pm25ml.collectors.export_pipeline import (
    ErrorWhileFetchingDataError,
    ExportPipeline,
    MissingDataError,
)
from pm25ml.logging import logger

if TYPE_CHECKING:
    from collections.abc import Collection

    from pm25ml.collectors.archived_file_validator import ArchivedFileValidator


class RawDataCollector:
    """Collects raw data from various sources to be uploaded to the archive."""

    def __init__(self, metadata_validator: ArchivedFileValidator) -> None:
        """
        Initialize the RawDataCollector with the metadata validator.

        :param metadata_validator: The validator to check if the results need to be uploaded.
        """
        self.metadata_validator = metadata_validator

    def collect(self, processors: Collection[ExportPipeline]) -> None:
        """
        Collect data from the given processors and upload them to the archive storage.

        :param processors: A collection of ExportPipeline instances to collect data from.
        """
        logger.info("Filtering down to only those datasets that need to be uploaded")
        # Now we only want to download the results if we never have before.
        # We can check with the archive_storage if the results already exist.
        # Evaluate which processors need upload in parallel for speed.
        filtered_processors = self._filter_processors_needing_upload(processors)

        logger.info(f"Go ahead and download {len(filtered_processors)} datasets")
        self._run_pipelines_in_parallel(filtered_processors)

        # Check all results were uploaded successfully, not just the ones we
        # downloaded this time.
        logger.info("Validating all recent results")
        self.metadata_validator.validate_all_results(
            [processor.get_config_metadata() for processor in filtered_processors],
        )

    def _filter_processors_needing_upload(
        self,
        processors: Collection[ExportPipeline],
    ) -> Collection[ExportPipeline]:
        with ThreadPoolExecutor() as executor:
            needs_upload_results = list(
                executor.map(
                    lambda processor: self.metadata_validator.needs_upload(
                        expected_result=processor.get_config_metadata(),
                    ),
                    processors,
                ),
            )

        return [
            processor
            for processor, needs_upload in zip(processors, needs_upload_results)
            if needs_upload
        ]

    def _run_pipelines_in_parallel(self, filtered_processors: Collection[ExportPipeline]) -> None:
        if not filtered_processors:
            logger.info("No processors to run, skipping upload.")
            return

        class _ResultStatus(Enum):
            SUCCESS = "success"
            MISSING_DATA = "missing_data"
            FAILURE = "failure"

        with ThreadPoolExecutor() as executor:

            def _upload_processor(
                processor: ExportPipeline,
            ) -> tuple[ExportPipeline, _ResultStatus, Exception | None]:
                result_subpath = processor.get_config_metadata().result_subpath
                try:
                    logger.info(
                        f"Starting upload for processor {result_subpath}",
                    )
                    processor.upload()
                except MissingDataError as e:
                    logger.warning(
                        f"Missing data for processor {result_subpath}",
                        exc_info=True,
                        stack_info=True,
                    )
                    return (processor, _ResultStatus.MISSING_DATA, e)
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"Failed to upload processor {result_subpath}",
                        exc_info=True,
                        stack_info=True,
                    )
                    return (processor, _ResultStatus.FAILURE, e)
                else:
                    return (processor, _ResultStatus.SUCCESS, None)

            results = executor.map(_upload_processor, filtered_processors)

            completed_results = list(results)

            results_by_status = {
                status: [(x[0], x[2]) for x in filter(lambda x: x[1] == status, completed_results)]
                for status in _ResultStatus.__members__.values()
            }

            if (
                results_by_status[_ResultStatus.FAILURE]
                or results_by_status[_ResultStatus.MISSING_DATA]
            ):
                failed_pipelines = results_by_status[_ResultStatus.FAILURE]
                missing_data_pipelines = results_by_status[_ResultStatus.MISSING_DATA]

                for pipeline, err in failed_pipelines:
                    result_subpath = pipeline.get_config_metadata().result_subpath
                    logger.error(
                        f"Failed to upload pipeline {result_subpath}",
                        exc_info=err,
                    )
                for pipeline, err in missing_data_pipelines:
                    result_subpath = pipeline.get_config_metadata().result_subpath
                    logger.warning(
                        f"Missing data for pipeline {result_subpath}",
                        exc_info=err,
                    )
                msg = (
                    f"Failed to upload {len(failed_pipelines)} pipelines and "
                    f"missing data for {len(missing_data_pipelines)} pipelines."
                )
                raise ErrorWhileFetchingDataError(msg)
