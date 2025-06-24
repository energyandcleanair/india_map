import os

import polars as pl
import pytest
from morefs.memory import MemFS

from pm25ml.collectors.pipeline_storage import IngestArchiveStorage

DESTINATION_BUCKET = "destination_bucket"


@pytest.fixture
def example_table():
    data = {
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
    }
    return pl.DataFrame(data)


@pytest.fixture
def in_memory_filesystem():
    return MemFS()


# Update test_write_to_destination to use a valid FileSystem
def test_write_to_destination(in_memory_filesystem, example_table) -> None:
    # Use LocalFileSystem pointing to the temp directory
    storage = IngestArchiveStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(example_table, "result_path")

    # Validate that the Parquet file was written to the temp directory
    result_path = os.path.join(DESTINATION_BUCKET, "result_path", "data.parquet")
    assert in_memory_filesystem.exists(result_path)


# Test for read_dataframe_metadata
def test_read_dataframe_metadata(in_memory_filesystem, example_table) -> None:
    storage = IngestArchiveStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    # Write the table to the destination
    storage.write_to_destination(example_table, "result_path")

    # Read the metadata
    metadata = storage.read_dataframe_metadata("result_path")

    # Validate the metadata
    assert metadata.num_rows == example_table.shape[0]
    assert metadata.num_columns == example_table.shape[1]


# Test for does_dataset_exist
def test_does_dataset_exist(in_memory_filesystem, example_table) -> None:
    storage = IngestArchiveStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    # Initially, the dataset should not exist
    assert not storage.does_dataset_exist("result_path")

    # Write the table to the destination
    storage.write_to_destination(example_table, "result_path")

    # Now, the dataset should exist
    assert storage.does_dataset_exist("result_path")
