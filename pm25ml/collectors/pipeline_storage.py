"""Feature planning for gridded feature collections."""

from fsspec import AbstractFileSystem
from polars import DataFrame

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

        with self.filesystem.open(parquet_file_path, "wb") as f:
            # Convert the DataFrame to Parquet format and write it to the file
            logger.info(f"Writing DataFrame to Parquet file at {parquet_file_path}")
            table.write_parquet(f)
