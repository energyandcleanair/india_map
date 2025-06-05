import os

import pyarrow as pa
import pyarrow.csv as pa_csv
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from pm25ml.collectors.gee.intermediate_storage import GeeIntermediateStorage

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
    pa_csv.write_csv(example_table, csv_buffer)
    return pa.BufferReader(csv_buffer.getvalue())


@pytest.fixture
def in_memory_filesystem():
    return MemoryFileSystem()


def test_get_intermediate_by_id(in_memory_filesystem, example_table) -> None:
    with in_memory_filesystem.open(os.path.join(INTERMEDIATE_BUCKET, "test_id.csv"), "wb") as f:
        pa_csv.write_csv(example_table, f)

    # Use LocalFileSystem pointing to the temp directory
    storage = GeeIntermediateStorage(
        filesystem=in_memory_filesystem,
        bucket=INTERMEDIATE_BUCKET,
    )

    table = storage.get_intermediate_by_id("test_id")

    assert table.equals(example_table)


def test_delete_intermediate_by_id(in_memory_filesystem, example_table) -> None:
    # Create a temporary file to simulate the intermediate file
    file_path = os.path.join(INTERMEDIATE_BUCKET, "test_id.csv")
    with in_memory_filesystem.open(file_path, "wb") as f:
        pa_csv.write_csv(example_table, f)

    storage = GeeIntermediateStorage(
        filesystem=in_memory_filesystem,
        bucket=INTERMEDIATE_BUCKET,
    )

    # Call the delete method
    storage.delete_intermediate_by_id("test_id")

    # Ensure the file no longer exists after deletion
    assert not in_memory_filesystem.exists(file_path)
