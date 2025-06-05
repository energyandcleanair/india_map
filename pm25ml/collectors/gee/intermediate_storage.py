"""Handles intermediate storage operations."""

from fsspec import AbstractFileSystem
from pyarrow import Table
from pyarrow.csv import ReadOptions, read_csv

from pm25ml.logging import logger


class GeeIntermediateStorage:
    """Handles the intermediate storage operations for the GeeExportPipeline."""

    def __init__(self, filesystem: AbstractFileSystem, bucket: str) -> None:
        """
        Initialize the GeeIntermediateStorage with the filesystem and bucket path.

        :param filesystem: The filesystem to use for reading and writing files.
        :param bucket: The bucket name where intermediate CSV files are stored.
        """
        self.filesystem = filesystem
        self.bucket = bucket

    def get_intermediate_by_id(self, file_id: str) -> Table:
        """Get intermediate data by ID."""
        csv_file_path = f"{self.bucket}/{file_id}.csv"
        logger.info(f"Reading intermediate CSV file from {csv_file_path}")
        with self.filesystem.open(csv_file_path) as file:
            read_options = ReadOptions(block_size=64 * 1024 * 1024)
            return read_csv(file, read_options=read_options)

    def delete_intermediate_by_id(self, file_id: str) -> None:
        """Delete intermediate data by ID."""
        csv_file_path = f"{self.bucket}/{file_id}.csv"
        logger.info(f"Deleting intermediate CSV file {csv_file_path}")
        self.filesystem.delete(csv_file_path)
