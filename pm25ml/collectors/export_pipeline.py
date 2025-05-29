from pm25ml.collectors.tasks import GriddingTask
from pm25ml.logging import logger
from pm25ml.run.download_results import INGEST_ARCHIVE_BUCKET_NAME


import pyarrow.fs as pafs
import pyarrow.parquet as pq
from pyarrow import Table
from pyarrow.csv import ReadOptions, read_csv


class GeeExportPipeline:
    def __init__(
        self, filesystem: pafs.GcsFileSystem, task: GriddingTask, result_subpath: str
    ):
        self.filesystem = filesystem
        self.task = task
        self.result_subpath = result_subpath

    def upload(self):
        logger.info(f"Task {self.task.description}: starting task")
        self.task.complete_task()
        logger.info(
            f"Task {self.task.description}: reading task result {self.task.result_bucket_path} CSV from GCS"
        )
        raw_table = self._read_csv_from_gcs()
        logger.info(f"Task {self.task.description}: processing raw CSV table for task")
        processed_table = self._process(raw_table)
        logger.info(
            f"Task {self.task.description}: writing task processed table to GCS {self.result_subpath}"
        )
        self._write_to_gcs(processed_table)
        logger.info(
            f"Task {self.task.description}: deleting task old CSV file from GCS {self.task.result_bucket_path}"
        )
        self._delete_csv_from_gcs()

    def _write_to_gcs(self, table):
        parquet_file_path = f"{INGEST_ARCHIVE_BUCKET_NAME}/{self.result_subpath}/"
        # Write Parquet dataset to GCS
        pq.write_to_dataset(
            table,
            root_path=parquet_file_path,
            filesystem=self.filesystem,
            basename_template="file{i}.parquet",
        )

    def _process(self, table: Table) -> Table:
        expected_columns = self.task.column_mappings.keys()

        # Ensure the table has the expected columns
        missing_columns = [
            col for col in expected_columns if col not in table.column_names
        ]
        if missing_columns:
            raise ValueError(
                f"Table is missing expected columns: {', '.join(missing_columns)}"
            )

        # Test that the columns aren't all null values
        columns_null_values = [
            col
            for col in expected_columns
            if table.column(col).null_count == table.num_rows
        ]
        if columns_null_values:
            raise ValueError(
                f"Table has columns with all null values: {', '.join(columns_null_values)}"
            )

        # Drop extra columns that are not in the expected columns
        extra_columns = [
            col for col in table.column_names if col not in expected_columns
        ]
        if extra_columns:
            logger.warning(
                f"Dropping extra columns from table: {', '.join(extra_columns)}"
            )
            table = table.drop(extra_columns)

        # Rename columns according to the mappings
        new_names = [
            self.task.column_mappings.get(col, col) for col in table.column_names
        ]
        table = table.rename_columns(new_names)

        # Sort the table (if possible) by preferred order
        preferred_sort_order = [
            "date",
            "grid_id",
        ]
        columns_to_sort = [
            col for col in preferred_sort_order if col in table.column_names
        ]
        if columns_to_sort:
            sort_orders = [(col, "descending") for col in columns_to_sort]
            table = table.sort_by(sort_orders)
        return table

    def _read_csv_from_gcs(self):
        csv_file_path = self.task.result_bucket_path
        without_prefix = csv_file_path.replace("gs://", "")

        # Read CSV from GCS as a dataset
        with self.filesystem.open_input_file(without_prefix) as csv_file:
            table = read_csv(csv_file, read_options=ReadOptions(use_threads=True))

        return table

    def _delete_csv_from_gcs(self):
        csv_file_path = self.task.result_bucket_path
        # Delete the CSV file from GCS
        self.filesystem.delete_file(csv_file_path)