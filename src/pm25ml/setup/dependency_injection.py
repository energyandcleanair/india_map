"""A module for setting up the PM2.5 ML project using Dependency Injection."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import arrow
import ee
import google.auth
from dependency_injector import containers, providers
from ee.featurecollection import FeatureCollection
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
from pm25ml.combiners.archive_wide_combiner import ArchiveWideCombiner
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.combiner import MonthlyCombiner
from pm25ml.logging import logger
from pm25ml.setup.pipelines import define_pipelines

LOCAL_GRID_ZIP_PATH = "./assets/grid_india_10km_shapefiles.zip"


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
        Path(LOCAL_GRID_ZIP_PATH),
    )


class Pm25mlContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for the PM2.5 ML project.

    This container manages the configuration and resources required for the project,
    """

    config = providers.Configuration(strict=True)

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
        MonthlyCombiner,
        combined_storage=combined_storage,
        archived_wide_combiner=archived_wide_combiner,
        months=config.months,
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
        years=config.years,
        months=config.months,
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

    container.config.gcp.gee.india_shapefile_asset.from_env("INDIA_SHAPEFILE_ASSET")

    container.config.start_month_as_str.from_env("START_MONTH")
    container.config.end_month_as_str.from_env("END_MONTH")

    container.config.start_month.from_value(
        arrow.get(container.config.start_month_as_str()),
    )
    container.config.end_month.from_value(
        arrow.get(container.config.end_month_as_str()),
    )
    container.config.years.from_value(
        list(
            range(
                container.config.start_month().year,
                container.config.end_month().year + 1,
            ),
        ),
    )
    container.config.months.from_value(
        list(
            arrow.Arrow.range(
                "month",
                start=container.config.start_month(),
                end=container.config.end_month(),
            ),
        ),
    )

    container.init_resources()
    return container
