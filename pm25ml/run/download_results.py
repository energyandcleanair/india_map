"""Runner to get the data from a variety of sources."""

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
from arrow import Arrow, get
from ee.featurecollection import FeatureCollection
from gcsfs import GCSFileSystem

from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.archived_file_validator import ArchivedFileValidator
from pm25ml.collectors.export_pipeline import ExportPipeline
from pm25ml.collectors.gee import GeeExportPipeline, GriddedFeatureCollectionPlanner
from pm25ml.collectors.gee.intermediate_storage import GeeIntermediateStorage
from pm25ml.collectors.grid_export_pipeline import GridExportPipeline
from pm25ml.collectors.grid_loader import load_grid_from_zip
from pm25ml.collectors.ned.coord_types import Lat, Lon
from pm25ml.collectors.ned.data_reader_merra import MerraDataReader
from pm25ml.collectors.ned.data_reader_omno2d import Omno2dReader
from pm25ml.collectors.ned.data_retriever_harmony import HarmonySubsetterDataRetriever
from pm25ml.collectors.ned.data_retriever_raw import RawEarthAccessDataRetriever
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.ned_export_pipeline import NedExportPipeline
from pm25ml.collectors.validate_configuration import VALID_COUNTRIES, validate_configuration
from pm25ml.combiners.archive_wide_combiner import ArchiveWideCombiner
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.logging import logger

GCP_PROJECT = os.environ["GCP_PROJECT"]
INDIA_SHAPEFILE_GEE_ASSET = os.environ["INDIA_SHAPEFILE_ASSET"]
CSV_BUCKET_NAME = os.environ["CSV_BUCKET_NAME"]
INGEST_ARCHIVE_BUCKET_NAME = os.environ["INGEST_ARCHIVE_BUCKET_NAME"]
COMBINED_BUCKET_NAME = os.environ["COMBINED_BUCKET_NAME"]

LOCAL_GRID_ZIP_PATH = "./grid_india_10km_shapefiles.zip"

MONTH_SHORT = "2023-01"
YEAR_SHORT = "2023"


def _main() -> None:
    ee.Authenticate()
    ee.Initialize(project=GCP_PROJECT)

    month_start = get(f"{MONTH_SHORT}-01")
    gee_india_grid_reference = FeatureCollection(INDIA_SHAPEFILE_GEE_ASSET)
    gee_india_grid_reference_size = gee_india_grid_reference.size().getInfo()
    if gee_india_grid_reference_size != VALID_COUNTRIES["india"]:
        msg = (
            f"Expected {VALID_COUNTRIES['india']} features in the GEE India grid, "
            f"but found {gee_india_grid_reference_size}."
        )
        raise ValueError(
            msg,
        )

    feature_planner = GriddedFeatureCollectionPlanner(grid=gee_india_grid_reference)

    gcs_filesystem = GCSFileSystem()

    intermediate_storage = GeeIntermediateStorage(
        filesystem=gcs_filesystem,
        bucket=CSV_BUCKET_NAME,
    )

    archive_storage = IngestArchiveStorage(
        filesystem=gcs_filesystem,
        destination_bucket=INGEST_ARCHIVE_BUCKET_NAME,
    )

    metadata_validator = ArchivedFileValidator(
        archive_storage=archive_storage,
    )

    month_end = month_start.shift(months=1).shift(days=-1)

    dates_in_month: list[Arrow] = list(
        Arrow.range("day", start=month_start, end=month_end),
    )

    gee_pipeline_constructor = GeeExportPipeline.with_storage(
        intermediate_storage=intermediate_storage,
        archive_storage=archive_storage,
    )

    in_memory_grid = load_grid_from_zip(Path(LOCAL_GRID_ZIP_PATH))

    ned_pipeline_constructor = NedExportPipeline.with_args(
        archive_storage=archive_storage,
        grid=in_memory_grid,
    )

    combined_storage = CombinedStorage(
        filesystem=gcs_filesystem,
        destination_bucket=COMBINED_BUCKET_NAME,
    )

    archived_wide_combiner = ArchiveWideCombiner(
        archive_storage=archive_storage,
        combined_storage=combined_storage,
    )

    bounds = in_memory_grid.bounds

    bounds_with_border = (
        Lon(bounds[0] - 1.0),
        Lat(bounds[1] - 1.0),
        Lon(bounds[2] + 1.0),
        Lat(bounds[3] + 1.0),
    )

    logger.info("Defining export pipelines datasets")
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
                selected_bands=["tropospheric_NO2_column_number_density"],
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
                    "leaf_area_index_high_vegetation_max",
                    "leaf_area_index_low_vegetation_max",
                    "leaf_area_index_high_vegetation_min",
                    "leaf_area_index_low_vegetation_min",
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
            result_subpath="country=india/dataset=srtm_elevation/type=static",
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
                year=int(YEAR_SHORT),
            ),
            result_subpath=f"country=india/dataset=modis_land_cover/year={YEAR_SHORT}",
        ),
        ned_pipeline_constructor.construct(
            dataset_descriptor=NedDatasetDescriptor(
                # https://disc.gsfc.nasa.gov/datasets/M2T1NXAER_5.12.4/summary
                dataset_name="M2T1NXAER",
                dataset_version="5.12.4",
                start_date=month_start,
                end_date=month_end,
                filter_bounds=bounds_with_border,
                variable_mapping={
                    "TOTEXTTAU": "aot",
                },
                level=None,
            ),
            dataset_reader=MerraDataReader(),
            dataset_retriever=HarmonySubsetterDataRetriever(),
            result_subpath=f"country=india/dataset=merra_aot/month={MONTH_SHORT}",
        ),
        ned_pipeline_constructor.construct(
            dataset_descriptor=NedDatasetDescriptor(
                # https://cmr.earthdata.nasa.gov/search/concepts/C1276812901-GES_DISC.html
                dataset_name="M2I3NVCHM",
                dataset_version="5.12.4",
                start_date=month_start,
                end_date=month_end,
                filter_bounds=bounds_with_border,
                variable_mapping={
                    "CO": "co",
                },
                level=-1,
            ),
            dataset_reader=MerraDataReader(),
            dataset_retriever=HarmonySubsetterDataRetriever(),
            result_subpath=f"country=india/dataset=merra_co/month={MONTH_SHORT}",
        ),
        ned_pipeline_constructor.construct(
            dataset_descriptor=NedDatasetDescriptor(
                # https://cmr.earthdata.nasa.gov/search/concepts/C1276812901-GES_DISC.html
                dataset_name="M2I3NVCHM",
                dataset_version="5.12.4",
                start_date=month_start,
                end_date=month_end,
                filter_bounds=bounds_with_border,
                variable_mapping={
                    "CO": "co",
                },
                level=0,
            ),
            dataset_reader=MerraDataReader(),
            dataset_retriever=HarmonySubsetterDataRetriever(),
            result_subpath=f"country=india/dataset=merra_co_top/month={MONTH_SHORT}",
        ),
        ned_pipeline_constructor.construct(
            dataset_descriptor=NedDatasetDescriptor(
                # https://cmr.earthdata.nasa.gov/searCch/concepts/C1266136111-GES_DISC.html
                dataset_name="OMNO2d",
                dataset_version="003",
                start_date=month_start,
                end_date=month_end,
                filter_bounds=bounds_with_border,
                variable_mapping={
                    "ColumnAmountNO2": "no2",
                },
                level=None,
            ),
            dataset_reader=Omno2dReader(),
            dataset_retriever=RawEarthAccessDataRetriever(),
            result_subpath=f"country=india/dataset=omi_no2/month={MONTH_SHORT}",
        ),
        ned_pipeline_constructor.construct(
            dataset_descriptor=NedDatasetDescriptor(
                # https://cmr.earthdata.nasa.gov/searCch/concepts/C1266136111-GES_DISC.html
                dataset_name="OMNO2d",
                dataset_version="004",
                start_date=month_start,
                end_date=month_end,
                filter_bounds=bounds_with_border,
                variable_mapping={
                    "ColumnAmountNO2": "no2",
                },
                level=None,
            ),
            dataset_reader=Omno2dReader(),
            dataset_retriever=RawEarthAccessDataRetriever(),
            result_subpath=f"country=india/dataset=omi_no2_v4/month={MONTH_SHORT}",
        ),
        GridExportPipeline(
            grid=in_memory_grid,
            archive_storage=archive_storage,
            result_subpath="country=india/dataset=grid/type=static",
        ),
    ]

    logger.info("Validating export pipeline config")
    validate_configuration(processors)

    logger.info("Filtering down to only those datasets that need to be uploaded")
    # Now we only want to download the results if we never have before.
    # We can check with the archive_storage if the results already exist.
    filtered_processors = [
        processor
        for processor in processors
        if metadata_validator.needs_upload(
            expected_result=processor.get_config_metadata(),
        )
    ]

    logger.info(f"Go ahead and download {len(filtered_processors)} datasets")
    _run_pipelines_in_parallel(filtered_processors)

    # Check all results were uploaded successfully, not just the ones we
    # downloaded this time.
    logger.info("Validating all recent and historical results")
    metadata_validator.validate_all_results(
        [processor.get_config_metadata() for processor in processors],
    )

    # Get files from the archive storage
    logger.info("Combining results from the archive storage")
    # This needs to be all processors, not just the filtered ones.
    archived_wide_combiner.combine(month=MONTH_SHORT, processors=processors)

    all_id_columns = {
        column for processor in processors for column in processor.get_config_metadata().id_columns
    }

    all_value_columns = {
        f"{processor.get_config_metadata().hive_path.require_key('dataset')}__{column}"
        for processor in processors
        for column in processor.get_config_metadata().value_columns
    }

    all_expected_columns = all_id_columns | all_value_columns

    expected_rows = VALID_COUNTRIES["india"] * len(dates_in_month)
    logger.info(
        f"Validating final combined result with {expected_rows} "
        f"expected rows and {len(all_expected_columns)} expected columns",
    )
    final_combined = combined_storage.read_dataframe(
        result_subpath=f"stage=combined_monthly/month={MONTH_SHORT}",
    )

    # Validate final combined result has expected rows and columns
    if final_combined.shape[0] != expected_rows:
        msg = (
            f"Expected {expected_rows} rows in the final combined result, "
            f"but found {final_combined.shape[0]} rows."
        )
        raise ValueError(msg)

    missing = all_expected_columns - set(final_combined.columns)
    if missing:
        msg = (
            f"Expected columns {all_expected_columns} in the final combined result, "
            f"but {missing} were missing."
        )
        raise ValueError(msg)


def _run_pipelines_in_parallel(filtered_processors: list[ExportPipeline]) -> None:
    with ThreadPoolExecutor() as executor:

        def _upload_processor(processor: ExportPipeline) -> None:
            try:
                processor.upload()
            except Exception:
                logger.error(
                    f"Failed to upload processor {processor.get_config_metadata().result_subpath}",
                    exc_info=True,
                    stack_info=True,
                )
                raise

        results = executor.map(_upload_processor, filtered_processors)

        deque(results, maxlen=0)


if __name__ == "__main__":
    _main()
