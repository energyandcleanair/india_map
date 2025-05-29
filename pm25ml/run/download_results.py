from arrow import Arrow, get
import ee

from gcsfs import GCSFileSystem

from ee import FeatureCollection

from pm25ml.collectors.pipeline_storage import GeeExportPipelineStorage
from pm25ml.collectors.feature_planner import (
    FeatureCollectionPlanner,
)
from pm25ml.logging import logger

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

    feature_planner = FeatureCollectionPlanner(grid=grid)

    gcs_filesystem = GCSFileSystem()

    gee_export_pipeline_storage = GeeExportPipelineStorage(
        filesystem=gcs_filesystem,
        intermediate_bucket=CSV_BUCKET_NAME,
        destination_bucket=INGEST_ARCHIVE_BUCKET_NAME,
    )

    month_end_exclusive = month_start.shift(months=1)

    dates_in_month: list[Arrow] = list(
        Arrow.range("day", start=month_start, end=month_end_exclusive)
    )

    pipeline_constructor = GeeExportPipeline.with_storage(
        gee_export_pipeline_storage=gee_export_pipeline_storage,
    )

    processors: list[GeeExportPipeline] = [
        pipeline_constructor.construct(
            plan=feature_planner.plan_grid_daily_average(
                collection_name="COPERNICUS/S5P/OFFL/L3_CO",
                selected_bands=["CO_column_number_density"],
                dates=dates_in_month,
            ),
            result_subpath=f"dataset=copernicus_s5p_co/month={MONTH_SHORT}",
        ),
        pipeline_constructor.construct(
            plan=feature_planner.plan_grid_daily_average(
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
        results = executor.map(lambda processor: processor.upload(), processors)

        try:
            for result in results:
                pass
        except Exception as e:
            logger.error(f"An error occurred during processing", exc_info=True, stack_info=True)
            raise e
        