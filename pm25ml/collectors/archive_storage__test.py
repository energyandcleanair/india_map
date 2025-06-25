import os

import polars as pl
from polars.testing import assert_frame_equal
import pytest
from morefs.memory import MemFS

from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.combiners.archive_wide_combiner import ArchiveWideCombiner
from pm25ml.combiners.combined_storage import CombinedStorage

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


def test__write_to_destination__valid_input__writes_parquet(
    in_memory_filesystem, example_table
) -> None:
    # Use LocalFileSystem pointing to the temp directory
    storage = IngestArchiveStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(example_table, "result_path")

    # Validate that the Parquet file was written to the temp directory
    result_path = os.path.join(DESTINATION_BUCKET, "result_path", "data.parquet")
    assert in_memory_filesystem.exists(result_path)


def test__read_dataframe_metadata__valid_input__returns_metadata(
    in_memory_filesystem, example_table
) -> None:
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


def test__does_dataset_exist__dataset_written__returns_true(
    in_memory_filesystem, example_table
) -> None:
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


def test__read_dataframe__valid_input__returns_dataframe(
    in_memory_filesystem, example_table
) -> None:
    storage = IngestArchiveStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    # Write the table to the destination
    storage.write_to_destination(example_table, "result_path")

    # Read the DataFrame
    data_asset = storage.read_data_asset("result_path")

    # Validate the DataFrame
    assert_frame_equal(data_asset.data_frame, example_table)


def test__read_data_asset__valid_input__returns_correct_hive_path(
    in_memory_filesystem, example_table
) -> None:
    storage = IngestArchiveStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    # Write the table to the destination
    storage.write_to_destination(example_table, "result_path")

    # Read the DataFrame
    data_asset = storage.read_data_asset("result_path")

    # Validate the HivePath
    assert data_asset.hive_path.result_subpath == "result_path"


def test__read_data_asset__valid_input__returns_correct_hive_path_with_dataset(
    in_memory_filesystem, example_table
) -> None:
    """Test that HivePath includes dataset metadata."""
    storage = IngestArchiveStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    # Write the table to the destination
    storage.write_to_destination(example_table, "dataset=test_dataset")

    # Read the DataFrame
    data_asset = storage.read_data_asset("dataset=test_dataset")

    # Validate the HivePath
    assert data_asset.hive_path.require_key("dataset") == "test_dataset"


def test__filter_paths_by_kv__valid_key_value__returns_filtered_paths(in_memory_filesystem) -> None:
    storage = IngestArchiveStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    # Create mock paths in the filesystem
    paths = [
        f"{DESTINATION_BUCKET}/key=value1/data.parquet",
        f"{DESTINATION_BUCKET}/key=value2/data.parquet",
        f"{DESTINATION_BUCKET}/key=value1/other/data.parquet",
    ]
    for path in paths:
        in_memory_filesystem.touch(path)

    # Filter paths by key-value pair
    filtered_paths = storage.filter_paths_by_kv("key", "value1")

    # Validate the filtered paths
    assert filtered_paths == ["key=value1", "key=value1/other"]
