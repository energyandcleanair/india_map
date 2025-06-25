"""Export pipeline for grid data to the ingest archive."""

from pm25ml.collectors.export_pipeline import ExportPipeline, PipelineConfig
from pm25ml.collectors.grid_loader import Grid
from pm25ml.collectors.pipeline_storage import IngestArchiveStorage


class GridExportPipeline(ExportPipeline):
    """A pipeline for exporting grid data to the ingest archive."""

    def __init__(
        self,
        grid: Grid,
        archive_storage: IngestArchiveStorage,
        result_subpath: str,
    ) -> None:
        """
        Initialize the GridExportPipeline with the grid and archive storage.

        Args:
            grid (Grid): The grid containing the data to be exported.
            archive_storage (IngestArchiveStorage): The storage where the results will be archived.
            result_subpath (str): The subpath where the results will be stored.

        """
        self.grid = grid
        self.archive_storage = archive_storage
        self.result_subpath = result_subpath

    def get_config_metadata(self) -> PipelineConfig:
        """
        Get the metadata for the grid export pipeline.

        Returns:
            ExportPipeline.Metadata: The metadata containing the result subpath and expected rows.

        """
        return PipelineConfig(
            result_subpath=self.result_subpath,
            expected_n_rows=self.grid.n_rows,
            id_columns={"grid_id"},
            value_columns={"lon", "lat"},
        )

    def upload(self) -> None:
        """Upload the grid data to the ingest archive."""
        # Select the grid_id, lon, and lat columns from the grid DataFrame
        table = self.grid.df.select(
            [self.grid.GRID_ID_COL, self.grid.LON_COL, self.grid.LAT_COL],
        )
        # Write the DataFrame to the destination bucket
        self.archive_storage.write_to_destination(
            table=table,
            result_subpath=self.result_subpath,
        )
