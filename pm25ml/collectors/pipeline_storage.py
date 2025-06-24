"""Feature planning for gridded feature collections."""

from typing import IO, cast

import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from polars import DataFrame
from pyarrow.parquet import FileMetaData

from pm25ml.logging import logger


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
            logger.info(f"Writing DataFrame to Parquet file at {parquet_file_path}")
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
