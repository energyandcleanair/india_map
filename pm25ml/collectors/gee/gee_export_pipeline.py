"""Export pipeline for Google Earth Engine data to the underlying storage."""

import contextlib
from time import sleep
from typing import TYPE_CHECKING

from ee.batch import Export, Task
from nanoid import generate
from polars import DataFrame

from pm25ml.collectors.export_pipeline import ExportPipeline
from pm25ml.logging import logger

from .feature_planner import FeaturePlan

if TYPE_CHECKING:
    from pm25ml.collectors.pipeline_storage import IngestArchiveStorage

    from .intermediate_storage import GeeIntermediateStorage


class GeeExportPipeline(ExportPipeline):
    """Handles the export of data from GEE to the specified storage."""

    @staticmethod
    def with_storage(
        *,
        archive_storage: "IngestArchiveStorage",
        intermediate_storage: "GeeIntermediateStorage",
    ) -> "GeePipelineConstructor":
        """
        Create a GeePipelineConstructor with the given storage.

        This allows for a more fluent interface when constructing pipelines.
        """
        return GeePipelineConstructor(
            archive_storage=archive_storage,
            intermediate_storage=intermediate_storage,
        )

    def __init__(
        self,
        *,
        intermediate_storage: "GeeIntermediateStorage",
        archive_storage: "IngestArchiveStorage",
        plan: FeaturePlan,
        result_subpath: str,
    ) -> None:
        """Initialize the GeeExportPipeline with the storage and plan."""
        self.archive_storage = archive_storage
        self.intermediate_storage = intermediate_storage
        self.plan = plan
        self.result_subpath = result_subpath

    def upload(self) -> None:
        """Upload the data from GEE to the underlying storage."""
        temporary_file_prefix = generate(size=10)
        task_name = f"{temporary_file_prefix}__{self.plan.feature_name}"[:100]

        # First, we define the task to export the data to GCS, run it, and then wait until it
        # completes.
        logger.info(f"Task {task_name}: starting task")
        task = self._define_task(
            task_name=task_name,
        )
        self._complete_task(task_name=task_name, task=task)

        # Now that the task is complete, we can read the CSV file from GCS.
        logger.info(f"Task {task_name}: reading task result CSV from GCS")
        raw_table = self.intermediate_storage.get_intermediate_by_id(task_name)

        # After reading the CSV file, we process it.
        logger.info(f"Task {task_name}: processing raw CSV table for task")
        processed_table = self._process(raw_table)

        # Then we write the processed table to the destination bucket format.
        logger.info(f"Task {task_name}: writing task processed table to GCS {self.result_subpath}")
        self.archive_storage.write_to_destination(processed_table, self.result_subpath)

        # Finally, we delete the temporary CSV file from the intermediate bucket. This should happen
        # in the future anyway with the bucket lifecycle, but we do it now to clean up the
        # intermediate storage.
        logger.info(f"Task {task_name}: deleting task old CSV file from GCS")
        self.intermediate_storage.delete_intermediate_by_id(task_name)

    def _define_task(self, task_name: str) -> Task:
        exported_properties = self.plan.intermediate_columns
        return Export.table.toCloudStorage(
            description=task_name,
            collection=self.plan.planned_collection,
            bucket=self.intermediate_storage.bucket,
            fileNamePrefix=task_name,
            fileFormat="CSV",
            selectors=exported_properties if not self.plan.ignore_selectors else None,
        )

    def _complete_task(self, *, task_name: str, task: Task) -> None:
        try:
            task.start()
            delay_backoff = 1.0
            growth_factor = 1.5
            max_delay = 10.0
            while task.active():
                logger.debug(
                    f"Task {task_name}: waiting for task to complete ({delay_backoff}s delay)",
                )
                sleep(delay_backoff)
                delay_backoff = min(max_delay, delay_backoff * growth_factor)

            if task.status().get("state") != "COMPLETED":
                error_message = task.status().get("error_message", "No error message")
                msg = f"Task {task_name} failed: {error_message}"
                raise RuntimeError(msg)
        finally:
            with contextlib.suppress(Exception):
                task.cancel()

    def _process(self, table: DataFrame) -> DataFrame:
        mappings = self.plan.column_mappings
        expected_columns = self.plan.intermediate_columns

        # Ensure the table has the expected columns
        missing_columns = [col for col in expected_columns if col not in table.columns]
        if missing_columns:
            msg = f"Table is missing expected columns: {', '.join(missing_columns)}"
            raise ValueError(msg)

        # Test that the columns aren't all null values
        columns_null_values = [
            col for col in expected_columns if table[col].null_count() == table.height
        ]
        if columns_null_values:
            msg = f"Table has columns with all null values: {', '.join(columns_null_values)}"
            raise ValueError(
                msg,
            )

        # Drop extra columns that are not in the expected columns
        extra_columns = [col for col in table.columns if col not in expected_columns]
        if extra_columns:
            logger.warning(f"Dropping extra columns from table: {', '.join(extra_columns)}")
            table = table.drop(extra_columns)

        # Rename columns according to the mappings
        table = table.rename(mappings)

        # Sort the table (if possible) by preferred order
        preferred_sort_order = [
            "date",
            "grid_id",
        ]
        columns_to_sort = [col for col in preferred_sort_order if col in table.columns]
        if columns_to_sort:
            table = table.sort(
                by=columns_to_sort,
            )
        return table


class GeePipelineConstructor:
    """
    A constructor for GeeExportPipeline that allows for a more fluent interface.

    Should be used with `GeeExportPipeline.with_storage()`.
    """

    def __init__(
        self,
        *,
        intermediate_storage: "GeeIntermediateStorage",
        archive_storage: "IngestArchiveStorage",
    ) -> None:
        """Initialize the GeePipelineConstructor with the storage."""
        self.archive_storage = archive_storage
        self.intermediate_storage = intermediate_storage

    def construct(
        self,
        plan: FeaturePlan,
        result_subpath: str,
    ) -> "GeeExportPipeline":
        """
        Construct a GeeExportPipeline with the given plan and result subpath.

        :param plan: The feature plan to use for the export.
        :param result_subpath: The subpath in the destination bucket where the results will be
        stored.
        """
        return GeeExportPipeline(
            archive_storage=self.archive_storage,
            intermediate_storage=self.intermediate_storage,
            plan=plan,
            result_subpath=result_subpath,
        )
