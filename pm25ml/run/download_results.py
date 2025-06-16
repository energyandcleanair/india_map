"""Runner to get the data from a variety of sources."""

import os
from concurrent.futures import ThreadPoolExecutor

import ee
from arrow import Arrow, get
from ee.featurecollection import FeatureCollection
from gcsfs import GCSFileSystem

from pm25ml.collectors.export_pipeline import ExportPipeline
from pm25ml.collectors.gee import GeeExportPipeline, GriddedFeatureCollectionPlanner
from pm25ml.collectors.gee.intermediate_storage import GeeIntermediateStorage
from pm25ml.collectors.grid_loader import load_grid_from_zip
from pm25ml.collectors.ned.coord_types import Lat, Lon
from pm25ml.collectors.ned.data_reader_merra import MerraDataReader
from pm25ml.collectors.ned.data_reader_omno2d import Omno2dReader
from pm25ml.collectors.ned.data_retriever_harmony import HarmonySubsetterDataRetriever
from pm25ml.collectors.ned.data_retriever_raw import RawEarthAccessDataRetriever
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.ned_export_pipeline import NedExportPipeline
from pm25ml.collectors.pipeline_storage import IngestArchiveStorage
from pm25ml.logging import logger

GCP_PROJECT = os.environ["GCP_PROJECT"]
INDIA_SHAPEFILE_GEE_ASSET = os.environ["INDIA_SHAPEFILE_ASSET"]
CSV_BUCKET_NAME = os.environ["CSV_BUCKET_NAME"]
INGEST_ARCHIVE_BUCKET_NAME = os.environ["INGEST_ARCHIVE_BUCKET_NAME"]

LOCAL_GRID_ZIP_PATH = "./grid_india_10km_shapefiles.zip"

MONTH_SHORT = "2023-01"

if __name__ == "__main__":
    ee.Authenticate()
    ee.Initialize(project=GCP_PROJECT)

    month_start = get(f"{MONTH_SHORT}-01")
    gee_grid_reference = FeatureCollection(INDIA_SHAPEFILE_GEE_ASSET)

    feature_planner = GriddedFeatureCollectionPlanner(grid=gee_grid_reference)

    gcs_filesystem = GCSFileSystem()

    intermediate_storage = GeeIntermediateStorage(
        filesystem=gcs_filesystem,
        bucket=CSV_BUCKET_NAME,
    )

    archive_storage = IngestArchiveStorage(
        filesystem=gcs_filesystem,
        destination_bucket=INGEST_ARCHIVE_BUCKET_NAME,
    )

    month_end_exclusive = month_start.shift(months=1)
    month_end_inclusive = month_end_exclusive.shift(days=-1)

    dates_in_month: list[Arrow] = list(
        Arrow.range("day", start=month_start, end=month_end_exclusive),
    )

    gee_pipeline_constructor = GeeExportPipeline.with_storage(
        intermediate_storage=intermediate_storage,
        archive_storage=archive_storage,
    )

    in_memory_grid = load_grid_from_zip(LOCAL_GRID_ZIP_PATH)

    ned_pipeline_constructor = NedExportPipeline.with_args(
        archive_storage=archive_storage,
        grid=in_memory_grid,
    )

    bounds_with_border = (
        Lon(in_memory_grid.total_bounds[0] - 1),
        Lat(in_memory_grid.total_bounds[1] - 1),
        Lon(in_memory_grid.total_bounds[2] + 1),
        Lat(in_memory_grid.total_bounds[3] + 1),
    )

    processors: list[ExportPipeline] = [
        gee_pipeline_constructor.construct(
            plan=feature_planner.plan_daily_average(
                collection_name="COPERNICUS/S5P/OFFL/L3_CO",
                selected_bands=["CO_column_number_density"],
                dates=dates_in_month,
            ),
            result_subpath=f"country=india/dataset=s5p_co/month={MONTH_SHORT}",
        ),
        gee_pipeline_constructor.construct(
            plan=feature_planner.plan_daily_average(
                collection_name="COPERNICUS/S5P/OFFL/L3_NO2",
                selected_bands=["NO2_column_number_density"],
                dates=dates_in_month,
            ),
            result_subpath=f"country=india/dataset=s5p_no2/month={MONTH_SHORT}",
        ),
        gee_pipeline_constructor.construct(
            plan=feature_planner.plan_daily_average(
                collection_name="ECMWF/ERA5_LAND/DAILY_AGGR",
                selected_bands=[
                    "temperature_2m",
                    "dewpoint_temperature_2m",
                    "u_component_of_wind_10m",
                    "v_component_of_wind_10m",
                    "total_precipitation_sum",
                    "surface_net_thermal_radiation_sum",
                    "surface_pressure",
                    "leaf_area_index_high_vegetation",
                    "leaf_area_index_low_vegetation",
                ],
                dates=dates_in_month,
            ),
            result_subpath=f"country=india/dataset=era5_land/month={MONTH_SHORT}",
        ),
        gee_pipeline_constructor.construct(
            plan=feature_planner.plan_daily_average(
                collection_name="MODIS/061/MCD19A2_GRANULES",
                selected_bands=["Optical_Depth_047", "Optical_Depth_055"],
                dates=dates_in_month,
            ),
            result_subpath=f"country=india/dataset=modis_aod/month={MONTH_SHORT}",
        ),
        gee_pipeline_constructor.construct(
            plan=feature_planner.plan_static_feature(
                image_name="USGS/SRTMGL1_003",
                selected_bands=["elevation"],
            ),
            result_subpath="country=india/dataset=srtm_elevation",
        ),
        gee_pipeline_constructor.construct(
            plan=feature_planner.plan_summarise_annual_classified_pixels(
                collection_name="MODIS/061/MCD12Q1",
                classification_band="LC_Type1",
                output_names_to_class_values={
                    "forest": [1, 2, 3, 4, 5],
                    "shrub": [6, 7],
                    "savanna": [9],
                    "urban": [13],
                    "water": [17],
                },
                year=2023,
            ),
            result_subpath="country=india/dataset=modis_land_cover/year=2023",
        ),
        gee_pipeline_constructor.construct(
            plan=feature_planner.plan_daily_average(
                collection_name="NASA/GSFC/MERRA/aer/2",
                selected_bands=["TOTEXTTAU"],
                dates=dates_in_month,
            ),
            result_subpath=f"country=india/dataset=merra_aot/month={MONTH_SHORT}",
        ),
        ned_pipeline_constructor.construct(
            dataset_descriptor=NedDatasetDescriptor(
                dataset_name="M2T1NXAER",
                dataset_version="5.12.4",
                start_date=month_start,
                end_date=month_end_inclusive,
                filter_bounds=bounds_with_border,
                source_variable_name="TOTEXTTAU",
                target_variable_name="merra_aot",
            ),
            dataset_reader=MerraDataReader(),
            dataset_retriever=HarmonySubsetterDataRetriever(),
            result_subpath=f"country=india/dataset=merra_aot/month={MONTH_SHORT}",
        ),
        ned_pipeline_constructor.construct(
            dataset_descriptor=NedDatasetDescriptor(
                dataset_name="M2I3NVCHM",
                dataset_version="5.12.4",
                start_date=month_start,
                end_date=month_end_inclusive,
                filter_bounds=bounds_with_border,
                source_variable_name="CO",
                target_variable_name="merra_co",
            ),
            dataset_reader=MerraDataReader(),
            dataset_retriever=HarmonySubsetterDataRetriever(),
            result_subpath=f"country=india/dataset=merra_co/month={MONTH_SHORT}",
        ),
        ned_pipeline_constructor.construct(
            dataset_descriptor=NedDatasetDescriptor(
                dataset_name="OMNO2d",
                dataset_version="003",
                start_date=month_start,
                end_date=month_end_inclusive,
                filter_bounds=bounds_with_border,
                source_variable_name="ColumnAmountNO2",
                target_variable_name="omi_no2",
            ),
            dataset_reader=Omno2dReader(),
            dataset_retriever=RawEarthAccessDataRetriever(),
            result_subpath=f"country=india/dataset=omi_no2/month={MONTH_SHORT}",
        ),
    ]

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda processor: processor.upload(), processors)

        try:
            for _ in results:
                pass
        except Exception:
            logger.error("An error occurred during processing", exc_info=True, stack_info=True)
            raise
