"""Feature planning for gridded feature collections."""

import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from pyarrow import Table
from pyarrow.csv import ReadOptions, read_csv

from pm25ml.logging import logger


class GeeExportPipelineStorage:
    """Handles the storage operations for the GeeExportPipeline."""

    def __init__(
        self,
        filesystem: AbstractFileSystem,
        intermediate_bucket: str,
        destination_bucket: str,
    ) -> None:
        """
        Initialize the GeeExportPipelineStorage with the filesystem and bucket paths.

        :param filesystem: The filesystem to use for reading and writing files.
        :param intermediate_bucket: The bucket name where intermediate CSV files are stored.
        :param destination_bucket: The bucket name where processed Parquet files will be written.
        """
        self.filesystem = filesystem
        self.intermediate_bucket = intermediate_bucket
        self.destination_bucket = destination_bucket

    def get_intermediate_by_id(self, file_id: str) -> Table:
        """
        Read the CSV file by ID from the intermediate bucket.

        :param file_id: The ID of the file to read.
        :return: A pyarrow Table containing the data from the CSV file.
        """
        csv_file_path = f"{self.intermediate_bucket}/{file_id}.csv"
        logger.info(f"Reading intermediate CSV file from {csv_file_path}")
        with self.filesystem.open(csv_file_path) as file:
            read_options = ReadOptions(block_size=64 * 1024 * 1024)
            return read_csv(file, read_options=read_options)  # type: ignore[arg-type]

    def delete_intermediate_by_id(self, file_id: str) -> None:
        """
        Delete the CSV file from the intermediate bucket by its ID.

        :param file_id: The ID of the file to delete.
        :return: None
        """
        csv_file_path = f"{self.intermediate_bucket}/{file_id}.csv"
        logger.info(f"Deleting intermediate CSV file {csv_file_path}")
        self.filesystem.delete(csv_file_path)

    def write_to_destination(self, table: Table, result_subpath: str) -> None:
        """
        Write the processed Table to the destination bucket.

        :param table: The pyarrow Table to write.
        :param result_subpath: The subpath in the destination bucket where the
        table will be written.
        """
        parquet_file_path = f"{self.destination_bucket}/{result_subpath}/"
        logger.info(f"Writing processed table to {parquet_file_path}")
        pq.write_to_dataset(
            table,
            root_path=parquet_file_path,
            filesystem=self.filesystem,
            basename_template="file{i}.parquet",
        )
