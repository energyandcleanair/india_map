"""Handles storage for combined data."""

from typing import IO, cast

import polars as pl
from fsspec import AbstractFileSystem
from polars import DataFrame

from pm25ml.logging import logger


class CombinedStorage:
    """Handles the storage operations for combined data."""

    def __init__(
        self,
        filesystem: AbstractFileSystem,
        destination_bucket: str,
    ) -> None:
        """
        Initialize the IngestArchiveStorage with the filesystem and bucket paths.

        :param filesystem: The filesystem to use for reading and writing files.
        :param destination_bucket: The bucket name where processed Parquet files will be written.
        """
        self.filesystem = filesystem
        self.destination_bucket = destination_bucket

    def write_to_destination(self, table: DataFrame, result_subpath: str) -> None:
        """
        Write the processed DataFrame to the destination bucket.

        :param table: The polars DataFrame to write.
        :param result_subpath: The subpath in the destination bucket where the
        table will be written.
        """
        parquet_file_path = f"{self.destination_bucket}/{result_subpath}/data.parquet"

        with self.filesystem.open(parquet_file_path, "wb") as file:
            # Convert the DataFrame to Parquet format and write it to the file
            logger.debug(f"Writing DataFrame to Parquet file at {parquet_file_path}")
            table.write_parquet(cast("IO[bytes]", file))

    def read_dataframe(
        self,
        result_subpath: str,
    ) -> DataFrame:
        """
        Read the processed DataFrame from the destination bucket.

        :param result_subpath: The subpath in the destination bucket where the
        DataFrame is stored.
        :return: The polars DataFrame read from the Parquet file.
        """
        parquet_file_path = f"{self.destination_bucket}/{result_subpath}/data.parquet"
        with self.filesystem.open(parquet_file_path) as file:
            return pl.read_parquet(cast("IO[bytes]", file))

    def does_dataset_exist(
        self,
        result_subpath: str,
    ) -> bool:
        """
        Check if the dataset exists in the destination bucket.

        :param result_subpath: The subpath in the destination bucket where the
        DataFrame is stored.
        :return: True if the dataset exists, False otherwise.
        """
        parquet_file_path = f"{self.destination_bucket}/{result_subpath}/data.parquet"
        return self.filesystem.exists(parquet_file_path)

    def scan_stage(
        self,
        stage: str,
    ) -> pl.LazyFrame:
        """
        Scan the specified stage in the destination bucket.

        :param stage: The stage to scan.
        :return: A LazyFrame representing the scanned data.
        """
        path = f"gs://{self.destination_bucket}/stage={stage}/"
        return pl.scan_parquet(
            path,
            hive_partitioning=True,
        )
