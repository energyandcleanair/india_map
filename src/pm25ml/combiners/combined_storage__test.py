import os
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
import pytest
from morefs.memory import MemFS

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
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(example_table, "result_path")

    # Validate that the Parquet file was written to the temp directory
    result_path = os.path.join(DESTINATION_BUCKET, "result_path", "data.parquet")
    assert in_memory_filesystem.exists(result_path)


def test__read_dataframe__valid_input__returns_dataframe(
    in_memory_filesystem, example_table
) -> None:
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
    )

    # Write the table to the destination
    storage.write_to_destination(example_table, "result_path")

    # Read the DataFrame
    dataframe = storage.read_dataframe("result_path")

    # Validate the DataFrame
    assert_frame_equal(dataframe, example_table)


def test__scan_stage__valid_stage__returns_lazyframe(in_memory_filesystem, example_table) -> None:
    with patch(
        "polars.scan_parquet",
    ) as mock_scan:
        storage = CombinedStorage(
            filesystem=in_memory_filesystem,
            destination_bucket=DESTINATION_BUCKET,
        )

        # Mock the scan_parquet method to return a LazyFrame
        mock_lazy_frame = pl.LazyFrame(example_table)
        mock_scan.return_value = mock_lazy_frame

        lazy_frame = storage.scan_stage("valid_stage")

        mock_scan.assert_called_once_with(
            f"gs://{DESTINATION_BUCKET}/stage=valid_stage/",
            hive_partitioning=True,
        )

        # Validate that the returned object is a LazyFrame
        assert lazy_frame is mock_lazy_frame
