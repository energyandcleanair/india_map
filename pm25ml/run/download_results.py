from time import sleep
from arrow import Arrow, get
import ee

import pyarrow.fs as pafs

from ee import FeatureCollection

from pm25ml.collectors.tasks import (
    TaskBuilder,
)

from pm25ml.collectors.export_pipeline import GeeExportPipeline
from concurrent.futures import ThreadPoolExecutor
import os

GCP_PROJECT = os.environ["GCP_PROJECT"]
INDIA_SHAPEFILE_ASSET = os.environ["INDIA_SHAPEFILE_ASSET"]
CSV_BUCKET_NAME = os.environ["CSV_BUCKET_NAME"]
INGEST_ARCHIVE_BUCKET_NAME = os.environ["INGEST_ARCHIVE_BUCKET_NAME"]

MONTH_SHORT = "2023-01"

if __name__ == "__main__":
    ee.Authenticate()
    ee.Initialize(project=GCP_PROJECT)

    month_start = get(f"{MONTH_SHORT}-01")
    grid = FeatureCollection(INDIA_SHAPEFILE_ASSET)

    builder = TaskBuilder(grid=grid, bucket_name=CSV_BUCKET_NAME)

    gcs_filesystem = pafs.GcsFileSystem()

    month_end_exclusive = month_start.shift(months=1)

    dates_in_month: list[Arrow] = list(
        Arrow.range("day", start=month_start, end=month_end_exclusive)
    )

    processors = [
        GeeExportPipeline(
            filesystem=gcs_filesystem,
            task=builder.build_grid_daily_average_task(
                collection_name="COPERNICUS/S5P/OFFL/L3_CO",
                selected_bands=["CO_column_number_density"],
                dates=dates_in_month,
            ),
            result_subpath=f"dataset=copernicus_s5p_co/month={MONTH_SHORT}",
        ),
        GeeExportPipeline(
            filesystem=gcs_filesystem,
            task=builder.build_grid_daily_average_task(
                collection_name="ECMWF/ERA5_LAND/DAILY_AGGR",
                selected_bands=[
                    "dewpoint_temperature_2m",
                    "surface_pressure",
                    "total_precipitation_sum",
                    "u_component_of_wind_10m",
                ],
                dates=dates_in_month,
            ),
            result_subpath=f"dataset=era5_land/month={MONTH_SHORT}",
        ),
    ]

    with ThreadPoolExecutor() as executor:
        executor.map(lambda processor: processor.upload(), processors)
