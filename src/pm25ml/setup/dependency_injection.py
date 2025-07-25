"""A module for setting up the PM2.5 ML project using Dependency Injection."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import arrow
import ee
import google.auth
import polars as pl
import requests
from dependency_injector import containers, providers
from ee.featurecollection import FeatureCollection
from fsspec.implementations.dirfs import DirFileSystem
from gcsfs import GCSFileSystem

from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.archived_file_validator import ArchivedFileValidator
from pm25ml.collectors.collector import RawDataCollector
from pm25ml.collectors.gee.feature_planner import GriddedFeatureCollectionPlanner
from pm25ml.collectors.gee.gee_export_pipeline import GeePipelineConstructor
from pm25ml.collectors.gee.intermediate_storage import GeeIntermediateStorage
from pm25ml.collectors.grid_loader import Grid, load_grid_from_zip
from pm25ml.collectors.ned.ned_export_pipeline import NedPipelineConstructor
from pm25ml.collectors.validate_configuration import VALID_COUNTRIES
from pm25ml.combiners.archive.combine_manager import MonthlyCombinerManager
from pm25ml.combiners.archive.combine_planner import CombinePlanner
from pm25ml.combiners.archive.combiner import ArchiveWideCombiner
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.recombiner.recombiner import Recombiner
from pm25ml.feature_generation.generate import FeatureGenerator
from pm25ml.imputation.spatial.daily_spatial_interpolator import DailySpatialInterpolator
from pm25ml.imputation.spatial.spatial_imputation_manager import SpatialImputationManager
from pm25ml.logging import logger
from pm25ml.sample.imputation_sampler import ImputationSamplerDefinition
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.model_references import build_training_ref
from pm25ml.setup.pipelines import define_pipelines
from pm25ml.setup.samplers import ImputationStep, define_samplers
from pm25ml.training.model_pipeline import ModelPipeline
from pm25ml.training.model_storage import ModelStorage

if TYPE_CHECKING:
    from collections.abc import Generator

    type BooleanSelector = Literal["true", "false"]

LOCAL_GRID_ZIP_PATH = Path("./assets/grid_india_10km_shapefiles.zip")
LOCAL_GRID_50KM_MAPPING_CSV_PATH = Path("./assets/grid_intersect_with_50km.csv")


@contextmanager
def _init_gee(
    gcp_project: str,
) -> Generator[None, None, None]:
    logger.debug("Initializing GEE with project: %s", gcp_project)
    creds, _ = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )
    ee.Initialize(project=gcp_project, credentials=creds)
    yield


def _load_india_grid_reference_asset(india_shapefile_asset: str) -> FeatureCollection:
    """
    Initialize the GeeIndiaGridReferenceResource with the GEE asset path.

    Args:
        india_shapefile_asset (str): The GEE asset path for the India shapefile.

    Returns:
        ee.FeatureCollection: The initialized FeatureCollection.

    """
    logger.debug("Loading India grid reference asset from: %s", india_shapefile_asset)
    gee_india_grid_reference = FeatureCollection(india_shapefile_asset)
    gee_india_grid_reference_size = gee_india_grid_reference.size().getInfo()
    if gee_india_grid_reference_size != VALID_COUNTRIES["india"]:
        msg = (
            f"Expected {VALID_COUNTRIES['india']} features in the GEE India grid, "
            f"but found {gee_india_grid_reference_size}."
        )
        raise ValueError(
            msg,
        )

    return gee_india_grid_reference


def _load_in_memory_grid() -> Grid:
    logger.debug("Loading in-memory grid from local zip file: %s", LOCAL_GRID_ZIP_PATH)
    return load_grid_from_zip(
        path_to_shapefile_zip=LOCAL_GRID_ZIP_PATH,
        path_to_50km_csv=LOCAL_GRID_50KM_MAPPING_CSV_PATH,
    )


class Pm25mlContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for the PM2.5 ML project.

    This container manages the configuration and resources required for the project,
    """

    config = providers.Configuration(strict=True)

    temporal_config = providers.Singleton(
        TemporalConfig,
        start_date=config.start_month,
        end_date=config.end_month,
    )

    gee_auth = providers.Resource(
        _init_gee,
        gcp_project=config.gcp.gcp_project,
    )

    gee_india_grid_reference = providers.Callable(
        _load_india_grid_reference_asset,
        india_shapefile_asset=config.gcp.gee.india_shapefile_asset,
    )

    feature_planner = providers.Singleton(
        GriddedFeatureCollectionPlanner,
        grid=gee_india_grid_reference,
    )

    local_rooted_filesystem = providers.Singleton(
        DirFileSystem,
        path="output",
    )

    gcs_filesystem: providers.Provider[GCSFileSystem] = providers.Singleton(
        GCSFileSystem,
    )

    intermediate_storage = providers.Singleton(
        GeeIntermediateStorage,
        filesystem=gcs_filesystem,
        bucket=config.gcp.csv_bucket,
    )

    archive_storage = providers.Singleton(
        IngestArchiveStorage,
        filesystem=gcs_filesystem,
        destination_bucket=config.gcp.archive_bucket,
    )

    metadata_validator = providers.Singleton(
        ArchivedFileValidator,
        archive_storage=archive_storage,
    )

    gee_pipeline_constructor = providers.Singleton(
        GeePipelineConstructor,
        archive_storage=archive_storage,
        intermediate_storage=intermediate_storage,
    )

    in_memory_grid = providers.Singleton(
        _load_in_memory_grid,
    )

    ned_pipeline_constructor = providers.Singleton(
        NedPipelineConstructor,
        archive_storage=archive_storage,
        grid=in_memory_grid,
    )

    combined_storage = providers.Singleton(
        CombinedStorage,
        filesystem=gcs_filesystem,
        destination_bucket=config.gcp.combined_bucket,
    )

    archived_wide_combiner = providers.Singleton(
        ArchiveWideCombiner,
        archive_storage=archive_storage,
        combined_storage=combined_storage,
    )

    monthly_combiner = providers.Singleton(
        MonthlyCombinerManager,
        combined_storage=combined_storage,
        archived_wide_combiner=archived_wide_combiner,
    )

    collector = providers.Singleton(
        RawDataCollector,
        metadata_validator=metadata_validator,
    )

    pipelines = providers.Singleton(
        define_pipelines,
        gee_pipeline_constructor=gee_pipeline_constructor,
        ned_pipeline_constructor=ned_pipeline_constructor,
        in_memory_grid=in_memory_grid,
        archive_storage=archive_storage,
        feature_planner=feature_planner,
        temporal_config=temporal_config,
    )

    combine_planner = providers.Singleton(
        CombinePlanner,
        temporal_config=temporal_config,
    )

    daily_spatial_interpolator = providers.Singleton(
        DailySpatialInterpolator,
        grid=in_memory_grid,
        value_column_regex_selector=config.spatial_computation_value_column_regex,
    )

    spatial_imputation_manager = providers.Singleton(
        SpatialImputationManager,
        combined_storage=combined_storage,
        spatial_imputer=daily_spatial_interpolator,
        temporal_config=temporal_config,
    )

    spatial_interpolation_recombiner = providers.Singleton(
        Recombiner,
        combined_storage=combined_storage,
        new_stage_name="combined_with_spatial_interpolation",
        temporal_config=temporal_config,
    )

    feature_generator = providers.Singleton(
        FeatureGenerator,
        combined_storage=combined_storage,
        recombiner=spatial_interpolation_recombiner,
        temporal_config=temporal_config,
    )

    imputation_samplers = providers.Singleton(
        define_samplers,
        combined_storage=combined_storage,
        temporal_config=temporal_config,
        imputation_steps=config.imputation_steps,
    )

    model_storage_backer = providers.Selector(
        config.local_training,
        true=local_rooted_filesystem,
        false=gcs_filesystem,
    )

    extra_sampler = providers.Selector(
        config.local_training,
        true=providers.Singleton(
            lambda x: x.filter(pl.col("month") == "2023-01").gather_every(100),
        ),
        false=providers.Singleton(lambda x: x),
    )

    model_store = providers.Singleton(
        ModelStorage,
        filesystem=model_storage_backer,
        bucket_name=config.gcp.model_storage_bucket,
    )

    ml_model_trainers = providers.Dict(
        aod=providers.Singleton(
            ModelPipeline,
            combined_storage=combined_storage,
            data_ref=build_training_ref("aod", extra_sampler),
            model_store=model_store,
            n_jobs=config.max_parallel_tasks,
        ),
    )


def init_dependencies_from_env() -> Pm25mlContainer:
    """
    Create a container instance with configuration loaded from environment variables.

    Returns:
        Container: An instance of the Container class with configuration set.

    """
    container = Pm25mlContainer()

    container.config.gcp.gcp_project.from_env("GCP_PROJECT")
    container.config.gcp.csv_bucket.from_env("CSV_BUCKET_NAME")
    container.config.gcp.archive_bucket.from_env("INGEST_ARCHIVE_BUCKET_NAME")
    container.config.gcp.combined_bucket.from_env("COMBINED_BUCKET_NAME")
    container.config.gcp.model_storage_bucket.from_env(
        "MODEL_STORAGE_BUCKET_NAME",
    )

    container.config.gcp.gee.india_shapefile_asset.from_env("INDIA_SHAPEFILE_ASSET")

    container.config.max_parallel_tasks.from_env(
        "MAX_PARALLEL_TASKS",
        as_=lambda x: int(x),
        default=str(os.cpu_count() or 1),
    )

    container.config.local_training.from_value(
        _parse_bool_env_var(
            os.getenv("LOCAL_TRAINING") or os.getenv("TINY_SAMPLE") or "false",
        ),
    )

    logger.info(f"Using local training: {container.config.local_training()}")

    container.config.start_month.from_env("START_MONTH", as_=lambda x: arrow.get(x, "YYYY-MM-DD"))
    container.config.end_month.from_env("END_MONTH", as_=lambda x: arrow.get(x, "YYYY-MM-DD"))

    container.config.spatial_computation_value_column_regex.from_env(
        "SPATIAL_COMPUTATION_VALUE_COLUMN_REGEX",
    )

    container.config.imputation_steps.from_value(
        [
            # AOD
            ImputationStep(
                imputation_sampler_definition=ImputationSamplerDefinition(
                    value_column="modis_aod__Optical_Depth_055",
                    model_name="aod",
                    percentage_sample=0.03,
                ),
            ),
            # Tropomi NO2
            ImputationStep(
                imputation_sampler_definition=ImputationSamplerDefinition(
                    value_column="s5p_no2__tropospheric_NO2_column_number_density",
                    model_name="no2",
                    percentage_sample=0.02,
                ),
            ),
            # Tropomi CO
            ImputationStep(
                imputation_sampler_definition=ImputationSamplerDefinition(
                    value_column="s5p_co__CO_column_number_density",
                    model_name="co",
                    percentage_sample=0.02,
                ),
            ),
        ],
    )

    container.init_resources()

    def try_getting_user_info() -> str | None:
        try:
            return requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
                headers={"Metadata-Flavor": "Google"},
                timeout=5,
            ).text
        except requests.RequestException as e:
            logger.debug(f"Failed to get user info: {e}")
            return None

    logger.debug(
        f"Google Cloud account from metadata server: {try_getting_user_info()}",
    )
    creds, _ = google.auth.default()
    service_account_email = getattr(creds, "service_account_email", None)
    logger.debug(
        f"Google Cloud account from auth: {service_account_email}",
    )

    return container


def _parse_bool_env_var(value: str) -> BooleanSelector:
    result = str(value).strip().lower() in ("1", "true", "yes", "on")
    return "true" if result else "false"
