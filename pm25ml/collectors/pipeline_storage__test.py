import os

import pyarrow as pa
import pyarrow.csv as pa_csv
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from pm25ml.collectors.pipeline_storage import IngestArchiveStorage

DESTINATION_BUCKET = "destination_bucket"


@pytest.fixture
def example_table():
    data = {
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
    }
    return pa.Table.from_pydict(data)


@pytest.fixture
def in_memory_filesystem():
    return MemoryFileSystem()


# Update test_write_to_destination to use a valid FileSystem
def test_write_to_destination(in_memory_filesystem, example_table) -> None:
    # Use LocalFileSystem pointing to the temp directory
    storage = IngestArchiveStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(example_table, "result_path")

    # Validate that the Parquet file was written to the temp directory
    result_path = os.path.join(DESTINATION_BUCKET, "result_path")
    assert in_memory_filesystem.exists(result_path)
