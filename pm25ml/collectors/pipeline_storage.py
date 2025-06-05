"""Feature planning for gridded feature collections."""

import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from pyarrow import Table

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
