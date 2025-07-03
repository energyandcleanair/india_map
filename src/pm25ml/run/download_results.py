"""Runner to get the data from a variety of sources."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import ee
import google.auth
from arrow import Arrow, get
from ee.featurecollection import FeatureCollection
from gcsfs import GCSFileSystem

from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.archived_file_validator import ArchivedFileValidator
from pm25ml.collectors.export_pipeline import (
    ErrorWhileFetchingDataError,
    ExportPipeline,
    MissingDataError,
)
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

if TYPE_CHECKING:
    from collections.abc import Iterable

GCP_PROJECT = os.environ["GCP_PROJECT"]
INDIA_SHAPEFILE_GEE_ASSET = os.environ["INDIA_SHAPEFILE_ASSET"]
CSV_BUCKET_NAME = os.environ["CSV_BUCKET_NAME"]
INGEST_ARCHIVE_BUCKET_NAME = os.environ["INGEST_ARCHIVE_BUCKET_NAME"]
COMBINED_BUCKET_NAME = os.environ["COMBINED_BUCKET_NAME"]

LOCAL_GRID_ZIP_PATH = "./assets/grid_india_10km_shapefiles.zip"

START_MONTH = get("2018-08-01")

END_MONTH = get("2025-03-31")


def _main() -> None:  # noqa: PLR0915
    creds, _ = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )
    ee.Initialize(project=GCP_PROJECT, credentials=creds)

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

    first_year = START_MONTH.year

    years = list(range(first_year, END_MONTH.year - 1))
    # Every month between START_MONTH and END_MONTH, inclusive.
    # We don't want to start from the first month of each year.
    months = list(
        Arrow.range(
            "month",
            start=START_MONTH,
            end=END_MONTH,
        ),
    )

    def _static_pipelines() -> Iterable[ExportPipeline]:
        """Fetch static datasets that do not change over time."""
        return [
            gee_pipeline_constructor.construct(
                plan=feature_planner.plan_static_feature(
                    image_name="USGS/SRTMGL1_003",
                    selected_bands=["elevation"],
                ),
                result_subpath="country=india/dataset=srtm_elevation/type=static",
            ),
            GridExportPipeline(
                grid=in_memory_grid,
                archive_storage=archive_storage,
                result_subpath="country=india/dataset=grid/type=static",
            ),
        ]

    def _yearly_pipelines(year: int) -> Iterable[ExportPipeline]:
        """Fetch datasets that are aggregated yearly."""
        return [
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
                    year=year,
                ),
                result_subpath=f"country=india/dataset=modis_land_cover/year={year}",
            ),
        ]

    def _monthly_pipelines(month_start: Arrow) -> Iterable[ExportPipeline]:
        """Fetch datasets that are aggregated monthly."""
        month_end = month_start.shift(months=1).shift(days=-1)

        dates_in_month: list[Arrow] = list(
            Arrow.range("day", start=month_start, end=month_end),
        )

        month_short = month_start.format("YYYY-MM")

        return [
            gee_pipeline_constructor.construct(
                plan=feature_planner.plan_daily_average(
                    collection_name="COPERNICUS/S5P/OFFL/L3_CO",
                    selected_bands=["CO_column_number_density"],
                    dates=dates_in_month,
                ),
                result_subpath=f"country=india/dataset=s5p_co/month={month_short}",
            ),
            gee_pipeline_constructor.construct(
                plan=feature_planner.plan_daily_average(
                    collection_name="COPERNICUS/S5P/OFFL/L3_NO2",
                    selected_bands=["tropospheric_NO2_column_number_density"],
                    dates=dates_in_month,
                ),
                result_subpath=f"country=india/dataset=s5p_no2/month={month_short}",
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
                result_subpath=f"country=india/dataset=era5_land/month={month_short}",
            ),
            gee_pipeline_constructor.construct(
                plan=feature_planner.plan_daily_average(
                    collection_name="MODIS/061/MCD19A2_GRANULES",
                    selected_bands=["Optical_Depth_047", "Optical_Depth_055"],
                    dates=dates_in_month,
                ),
                result_subpath=f"country=india/dataset=modis_aod/month={month_short}",
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
                result_subpath=f"country=india/dataset=merra_aot/month={month_short}",
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
                result_subpath=f"country=india/dataset=merra_co/month={month_short}",
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
                result_subpath=f"country=india/dataset=merra_co_top/month={month_short}",
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
                    interpolation_method="linear",
                ),
                dataset_reader=Omno2dReader(),
                dataset_retriever=RawEarthAccessDataRetriever(),
                result_subpath=f"country=india/dataset=omi_no2/month={month_short}",
            ),
        ]

    yearly_pipelines = [pipeline for year in years for pipeline in _yearly_pipelines(year)]

    monthly_pipelines = [pipeline for month in months for pipeline in _monthly_pipelines(month)]

    static_pipelines = _static_pipelines()

    logger.info("Defining export pipelines datasets")
    processors: list[ExportPipeline] = [
        *reversed(yearly_pipelines),
        *reversed(monthly_pipelines),
        *static_pipelines,
    ]

    logger.info("Validating export pipeline config")
    validate_configuration(processors)

    logger.info("Filtering down to only those datasets that need to be uploaded")
    # Now we only want to download the results if we never have before.
    # We can check with the archive_storage if the results already exist.
    # Evaluate which processors need upload in parallel for speed.
    filtered_processors = _filter_processors_needing_upload(metadata_validator, processors)

    logger.info(f"Go ahead and download {len(filtered_processors)} datasets")
    _run_pipelines_in_parallel(filtered_processors)

    # Check all results were uploaded successfully, not just the ones we
    # downloaded this time.
    logger.info("Validating all recent results")
    metadata_validator.validate_all_results(
        [processor.get_config_metadata() for processor in filtered_processors],
    )

    # Get files from the archive storage
    logger.info("Combining results from the archive storage")

    for month in months:
        month_short = month.format("YYYY-MM")

        dates_in_month: list[Arrow] = list(
            Arrow.range("day", start=month, end=month.shift(months=1).shift(days=-1)),
        )

        if archived_wide_combiner.needs_combining(
            month=month_short,
        ):
            # This needs to be all processors, not just the filtered ones.
            archived_wide_combiner.combine(month=month_short, processors=processors)

        all_id_columns = {
            column
            for processor in processors
            for column in processor.get_config_metadata().id_columns
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
            result_subpath=f"stage=combined_monthly/month={month_short}",
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


def _filter_processors_needing_upload(
    metadata_validator: ArchivedFileValidator,
    processors: list[ExportPipeline],
) -> list[ExportPipeline]:
    with ThreadPoolExecutor() as executor:
        needs_upload_results = list(
            executor.map(
                lambda processor: metadata_validator.needs_upload(
                    expected_result=processor.get_config_metadata(),
                ),
                processors,
            ),
        )

    return [
        processor
        for processor, needs_upload in zip(processors, needs_upload_results)
        if needs_upload
    ]


def _run_pipelines_in_parallel(filtered_processors: list[ExportPipeline]) -> None:
    class _ResultStatus(Enum):
        SUCCESS = "success"
        MISSING_DATA = "missing_data"
        FAILURE = "failure"

    with ThreadPoolExecutor() as executor:

        def _upload_processor(
            processor: ExportPipeline,
        ) -> tuple[ExportPipeline, _ResultStatus, Exception | None]:
            try:
                processor.upload()
            except MissingDataError as e:
                logger.warning(
                    f"Missing data for processor {processor.get_config_metadata().result_subpath}",
                    exc_info=True,
                    stack_info=True,
                )
                return (processor, _ResultStatus.MISSING_DATA, e)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"Failed to upload processor {processor.get_config_metadata().result_subpath}",
                    exc_info=True,
                    stack_info=True,
                )
                return (processor, _ResultStatus.FAILURE, e)
            else:
                return (processor, _ResultStatus.SUCCESS, None)

        results = executor.map(_upload_processor, filtered_processors)

        completed_results = list(results)

        results_by_status = {
            status: [(x[0], x[2]) for x in filter(lambda x: x[1] == status, completed_results)]
            for status in _ResultStatus.__members__.values()
        }

        if results_by_status[_ResultStatus.FAILURE] or not results_by_status[_ResultStatus.SUCCESS]:
            failed_pipelines = results_by_status[_ResultStatus.FAILURE]
            missing_data_pipelines = results_by_status[_ResultStatus.MISSING_DATA]

            for pipeline, err in failed_pipelines:
                logger.error(
                    f"Failed to upload pipeline {pipeline.get_config_metadata().result_subpath}",
                    exc_info=err,
                )
            for pipeline, err in missing_data_pipelines:
                logger.warning(
                    f"Missing data for pipeline {pipeline.get_config_metadata().result_subpath}",
                    exc_info=err,
                )
            msg = (
                f"Failed to upload {len(failed_pipelines)} pipelines and "
                f"missing data for {len(missing_data_pipelines)} pipelines."
            )
            raise ErrorWhileFetchingDataError(msg)


if __name__ == "__main__":
    _main()
