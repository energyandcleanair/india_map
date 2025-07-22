import pytest
from polars.testing import assert_frame_equal
from polars import DataFrame
import polars as pl
from morefs.memory import MemFS

from pm25ml.collectors.grid_loader import Grid
from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.misc.grid_export_pipeline import GridExportPipeline


@pytest.fixture
def mock_archive_storage():
    filesystem = MemFS()
    destination_bucket = "mock_bucket"
    return IngestArchiveStorage(filesystem=filesystem, destination_bucket=destination_bucket)


@pytest.fixture
def mock_grid():
    return Grid(
        DataFrame(
            {
                "grid_id": ["01", "02", "03", "04"],
                "lon": [0.5, 1.5, 2.5, 3.5],
                "lat": [2.5, 3.5, 4.5, 5.5],
                "extra_col": [10, 20, 30, 40],
            }
        )
    )


def test__GridExportPipeline__get_config_metadata__returns_correct_metadata(
    mock_grid, mock_archive_storage
):
    pipeline = GridExportPipeline(
        grid=mock_grid,
        archive_storage=mock_archive_storage,
        result_subpath="mock/subpath",
    )

    metadata = pipeline.get_config_metadata()

    assert metadata.result_subpath == "mock/subpath"
    assert metadata.expected_rows == mock_grid.n_rows
    assert metadata.id_columns == {"grid_id"}
    assert metadata.value_columns == {"lon", "lat"}


def test__GridExportPipeline__upload__writes_correct_data(mock_grid, mock_archive_storage):
    pipeline = GridExportPipeline(
        grid=mock_grid,
        archive_storage=mock_archive_storage,
        result_subpath="mock/subpath",
    )

    pipeline.upload()

    written_table = pl.read_parquet(
        mock_archive_storage.filesystem.open("mock_bucket/mock/subpath/data.parquet")
    )

    expected_data = DataFrame(
        {
            "grid_id": ["01", "02", "03", "04"],
            "lon": [0.5, 1.5, 2.5, 3.5],
            "lat": [2.5, 3.5, 4.5, 5.5],
        }
    )

    assert_frame_equal(
        written_table, expected_data, check_row_order=False, check_column_order=False
    )
