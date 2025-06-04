import os

import pyarrow as pa
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from pm25ml.collectors.pipeline_storage import GeeExportPipelineStorage

INTERMEDIATE_BUCKET = "intermediate_bucket"
DESTINATION_BUCKET = "destination_bucket"


@pytest.fixture
def example_table():
    data = {
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
    }
    return pa.Table.from_pydict(data)


# Update test_get_intermediate_by_id to use a valid NativeFile
@pytest.fixture
def mock_csv_file(example_table):
    # Create a mock CSV file from the example table
    csv_buffer = pa.BufferOutputStream()
    pa.csv.write_csv(example_table, csv_buffer)
    return pa.BufferReader(csv_buffer.getvalue())


@pytest.fixture
def in_memory_filesystem():
    return MemoryFileSystem()


def test_get_intermediate_by_id(in_memory_filesystem, example_table) -> None:
    with in_memory_filesystem.open(os.path.join(INTERMEDIATE_BUCKET, "test_id.csv"), "wb") as f:
        pa.csv.write_csv(example_table, f)

    # Use LocalFileSystem pointing to the temp directory
    storage = GeeExportPipelineStorage(
        filesystem=in_memory_filesystem,
        intermediate_bucket=INTERMEDIATE_BUCKET,
        destination_bucket=DESTINATION_BUCKET,
    )

    table = storage.get_intermediate_by_id("test_id")

    assert table.equals(example_table)


# Update test_write_to_destination to use a valid FileSystem
def test_write_to_destination(in_memory_filesystem, example_table) -> None:
    # Use LocalFileSystem pointing to the temp directory
    storage = GeeExportPipelineStorage(
        filesystem=in_memory_filesystem,
        intermediate_bucket=INTERMEDIATE_BUCKET,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(example_table, "result_path")

    # Validate that the Parquet file was written to the temp directory
    result_path = os.path.join(DESTINATION_BUCKET, "result_path")
    assert in_memory_filesystem.exists(result_path)


def test_delete_intermediate_by_id(in_memory_filesystem, example_table) -> None:
    # Create a temporary file to simulate the intermediate file
    file_path = os.path.join(INTERMEDIATE_BUCKET, "test_id.csv")
    with in_memory_filesystem.open(file_path, "wb") as f:
        pa.csv.write_csv(example_table, f)

    storage = GeeExportPipelineStorage(
        filesystem=in_memory_filesystem,
        intermediate_bucket=INTERMEDIATE_BUCKET,
        destination_bucket=DESTINATION_BUCKET,
    )

    # Call the delete method
    storage.delete_intermediate_by_id("test_id")

    # Ensure the file no longer exists after deletion
    assert not in_memory_filesystem.exists(file_path)
