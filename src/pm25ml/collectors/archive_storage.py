"""Handles storage for the export pipelines."""

from dataclasses import dataclass
from typing import IO, cast

import polars as pl
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from polars import DataFrame
from pyarrow.parquet import FileMetaData

from pm25ml.hive_path import HivePath
from pm25ml.logging import logger


@dataclass
class IngestDataAsset:
    """Represents an ingest data asset with its metadata and path."""

    data_frame: DataFrame
    hive_path: HivePath


class IngestArchiveStorage:
    """Handles the storage operations for the export pipeline."""

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

    def read_dataframe_metadata(
        self,
        result_subpath: str,
    ) -> FileMetaData:
        """
        Read the metadata DataFrame from the destination bucket.

        :param result_subpath: The subpath in the destination bucket where the
        metadata DataFrame is stored.
        :return: The polars DataFrame containing metadata.
        """
        parquet_file_path = f"{self.destination_bucket}/{result_subpath}/data.parquet"

        parquet_file = pq.ParquetFile(parquet_file_path, filesystem=self.filesystem)
        return parquet_file.metadata

    def read_data_asset(
        self,
        result_subpath: str,
    ) -> IngestDataAsset:
        """
        Read the processed DataFrame from the destination bucket.

        :param result_subpath: The subpath in the destination bucket where the
        DataFrame is stored.
        :return: The IngestDataAsset containing the DataFrame and its HivePath.
        """
        parquet_file_path = f"{self.destination_bucket}/{result_subpath}/data.parquet"
        hive_path = HivePath(f"{result_subpath}")
        with self.filesystem.open(parquet_file_path) as file:
            data_frame = pl.read_parquet(cast("IO[bytes]", file))
            return IngestDataAsset(data_frame=data_frame, hive_path=hive_path)

    def does_dataset_exist(
        self,
        result_subpath: str,
    ) -> bool:
        """
        Check if the dataset exists in the destination bucket.

        :param result_subpath: The subpath in the destination bucket to check.
        :return: True if the dataset exists, False otherwise.
        """
        parquet_file_path = f"{self.destination_bucket}/{result_subpath}/data.parquet"
        return self.filesystem.exists(parquet_file_path)

    def filter_paths_by_kv(self, key: str, value: str) -> list[str]:
        """
        Filter paths in the destination bucket based on a key-value pair.

        :param key: The key to filter by.
        :param value: The value to filter by.

        :return: A list of paths that match the key-value pair.
        """
        glob = f"{self.destination_bucket}/**/{key}={value}/**/data.parquet"
        logger.debug(f"Filtering paths with glob pattern: {glob}")

        results = cast("list[str]", self.filesystem.glob(glob))

        # Filter results so that they don't have the prefix of the destination bucket
        # and suffix of /data.parquet
        return [
            result.replace(f"{self.destination_bucket}/", "")
            .lstrip("/")
            .replace("/data.parquet", "")
            .rstrip("/")
            for result in results
        ]
