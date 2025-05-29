from pm25ml.logging import logger


import pyarrow.fs as pafs
import pyarrow.parquet as pq
from pyarrow import Table
from pyarrow.csv import ReadOptions, read_csv


class GeeExportPipelineStorage:
    def __init__(
        self,
        filesystem: pafs.GcsFileSystem,
        intermediate_bucket: str,
        destination_bucket: str,
    ):
        self.filesystem = filesystem
        self.intermediate_bucket = intermediate_bucket
        self.destination_bucket = destination_bucket

    def get_intermediate_by_id(self, id: str) -> Table:
        """
        Reads a CSV file from the intermediate bucket by its ID and returns it as a Table.
        """
        csv_file_path = f"{self.intermediate_bucket}/{id}.csv"
        logger.info(f"Reading intermediate CSV file from {csv_file_path}")
        with self.filesystem.open_input_file(csv_file_path) as csv_file:
            table = read_csv(csv_file, read_options=ReadOptions(use_threads=True))
        return table

    def delete_intermediate_by_id(self, id: str):
        """
        Deletes a CSV file from the intermediate bucket by its ID.
        """
        csv_file_path = f"{self.intermediate_bucket}/{id}.csv"
        logger.info(f"Deleting intermediate CSV file {csv_file_path}")
        self.filesystem.delete_file(csv_file_path)

    def write_to_destination(self, table: Table, result_subpath: str):
        """
        Writes the processed Table to the destination bucket in Parquet format.
        """
        parquet_file_path = f"{self.destination_bucket}/{result_subpath}/"
        logger.info(f"Writing processed table to {parquet_file_path}")
        pq.write_to_dataset(
            table,
            root_path=parquet_file_path,
            filesystem=self.filesystem,
            basename_template="file{i}.parquet",
        )