import contextlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

import arrow
import ee
import google.auth.impersonated_credentials
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pm25ml.collectors.gee.feature_planner import FeaturePlan, GriddedFeatureCollectionPlanner

import google.auth

pytestmark = pytest.mark.integration

BUCKET_NAME = os.environ.get("IT_GEE_ASSET_BUCKET_NAME")
GEE_IT_ASSET_ROOT = os.environ.get("IT_GEE_ASSET_ROOT")

GEE_IMAGE_COLLECTION_ROOT = f"{GEE_IT_ASSET_ROOT}/dummy_data"
GEE_GRID_LOCATION = f"{GEE_IT_ASSET_ROOT}/grid"

ASSET_DIR = Path("pm25ml", "collectors", "gee", "feature_planner__it_assets")

@pytest.fixture(scope="module", autouse=True)
def check_env():
    if not BUCKET_NAME:
        raise ValueError(
            "Environment variable IT_GEE_ASSET_BUCKET_NAME must be set for integration tests.",
        )
    if not GEE_IT_ASSET_ROOT:
        raise ValueError(
            "Environment variable IT_GEE_ASSET_ROOT must be set for integration tests.",
        )

@pytest.fixture(scope="module")
def initialize_gee():
    ee.Initialize()

    check_ee_initialised()

def check_ee_initialised() -> None:
    ee.data.getAssetRoots()

@pytest.fixture(scope="module", autouse=True, )
def upload_dummy_tiffs(initialize_gee) -> dict[str, str]:
    # We define these files manually up front as it's easier to manage than to
    # read from the directories and pull metadata out.
    assets_to_upload = [
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2022-12-30.tif"),
            "date": datetime(2022, 12, 30, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/dummy_data/dummy_image_2022-12-30.tif",
            "bucket_path": "dummy_data/dummy_image_2022-12-30.tif",
            "asset_path": f"{GEE_IMAGE_COLLECTION_ROOT}/2022-12-30",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2022-12-31.tif"),
            "date": datetime(2022, 12, 31, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/dummy_data/dummy_image_2022-12-31.tif",
            "bucket_path": "dummy_data/dummy_image_2022-12-31.tif",
            "asset_path": f"{GEE_IMAGE_COLLECTION_ROOT}/2022-12-31",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2023-01-01.tif"),
            "date": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/dummy_data/dummy_image_2023-01-01.tif",
            "bucket_path": "dummy_data/dummy_image_2023-01-01.tif",
            "asset_path": f"{GEE_IMAGE_COLLECTION_ROOT}/2023-01-01",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2023-01-02.tif"),
            "date": datetime(2023, 1, 2, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/dummy_data/dummy_image_2023-01-02.tif",
            "bucket_path": "dummy_data/dummy_image_2023-01-02.tif",
            "asset_path": f"{GEE_IMAGE_COLLECTION_ROOT}/2023-01-02",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "test-grid.zip"),
            "gcs_path": f"gs://{BUCKET_NAME}/dummy_data/test-grid.zip",
            "bucket_path": "dummy_data/test-grid.zip",
            "asset_path": GEE_GRID_LOCATION,
            "type": "table",
        },
    ]

    upload_to_gcs(assets_to_upload)
    upload_to_ee(assets_to_upload)

    return {
        "image_collection": GEE_IMAGE_COLLECTION_ROOT,
        "grid": GEE_GRID_LOCATION,
    }


def upload_to_gcs(
    assets: list,
):
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Empty the bucket before uploading new files
    blobs = bucket.list_blobs(prefix="")
    for blob in blobs:
        blob.delete()

    for upload in assets:
        blob = bucket.blob(upload["bucket_path"])
        blob.upload_from_filename(upload["local_path"])


def upload_to_ee(
    assets_to_upload: list,
):
    import ee
    import ee.data


    # First we need to delete any existing assets in the root folder
    def delete_assets_recursively(asset_id):
        """Recursively delete all assets under the given asset_id."""
        assets = ee.data.listAssets({"parent": asset_id})
        for asset in assets["assets"]:
            if asset["type"] == "FOLDER" or asset["type"] == "IMAGE_COLLECTION":
                delete_assets_recursively(asset["id"])
            else:
                ee.data.deleteAsset(asset["id"])

        ee.data.deleteAsset(asset_id)

    with contextlib.suppress(Exception):
        delete_assets_recursively(GEE_IT_ASSET_ROOT)

    # Then we need to create the folders and collections we need.
    ee.data.createAsset({"type": "FOLDER"}, GEE_IT_ASSET_ROOT)
    ee.data.createAsset({"type": "IMAGE_COLLECTION"}, GEE_IMAGE_COLLECTION_ROOT)

    tasks = []

    # Then we can start the ingestion tasks for each asset.
    for upload in assets_to_upload:
        if upload["type"] == "image":
            task = ee.data.startIngestion(
                None,
                {
                    "name": upload["asset_path"],
                    "tilesets": [
                        {
                            "sources": [
                                {
                                    "uris": [upload["gcs_path"]],
                                },
                            ],
                        },
                    ],
                    "startTime": upload["date"].isoformat(),
                    "endTime": (upload["date"] + timedelta(days=1)).isoformat(),
                },
            )
        elif upload["type"] == "table":
            task = ee.data.startTableIngestion(
                None,
                {
                    "name": upload["asset_path"],
                    "sources": [
                        {
                            "uris": [upload["gcs_path"]],
                        },
                    ],
                },
            )

        tasks.append(task)

    # Finally, we can wait for the tasks to complete.
    incomplete_task_ids = [task["id"] for task in tasks]
    while incomplete_task_ids:
        for task in incomplete_task_ids[:]:
            status = ee.data.getTaskStatus(task)
            if status and status[0]["state"] == "COMPLETED":
                incomplete_task_ids.remove(task)
            elif status and status[0]["state"] == "FAILED":
                raise RuntimeError(f"Task {task} failed: {status[0]['error_message']}")
        if incomplete_task_ids:
            print(f"Waiting for {len(incomplete_task_ids)} tasks to complete...")
            sleep(5)


@pytest.fixture(scope="module")
def feature_planner(upload_dummy_tiffs):
    return GriddedFeatureCollectionPlanner(
        grid=ee.FeatureCollection(upload_dummy_tiffs["grid"]),
    )


def test_plan_daily_average(
    feature_planner: GriddedFeatureCollectionPlanner,
    upload_dummy_tiffs,
):
    daily_average_plan = feature_planner.plan_daily_average(
        collection_name=upload_dummy_tiffs["image_collection"],
        selected_bands=["b1", "b2"],
        dates=[
            arrow.get("2023-01-01"),
            arrow.get("2023-01-02"),
        ],
    )

    actual_df = execute_plan_to_dataframe(daily_average_plan)

    first_date = arrow.get("2023-01-01").date()
    second_date = arrow.get("2023-01-02").date()

    expected_df = pl.DataFrame(
        {
            "date": [
                first_date,
                first_date,
                first_date,
                first_date,
                second_date,
                second_date,
                second_date,
                second_date,
            ],
            "b1_mean": [
                14.5,
                14.5,
                6.5,
                6.5,
                15.5,
                15.5,
                7.5,
                7.5,
            ],
            "b2_mean": [
                114.5,
                114.5,
                106.5,
                106.5,
                115.5,
                115.5,
                107.5,
                107.5,
            ],
            "grid_id": [
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
            ],
        },
    )

    assert_frame_equal(
        actual_df,
        expected_df,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        rtol=0,
        atol=0.2,
    )


def execute_plan_to_dataframe(feature_plan: FeaturePlan):
    """Extract features from the planned collection into a Polars DataFrame."""

    def convert_record(record):
        # Extract the date and grid_id from the record and then the any remaining properties
        date = arrow.get(record["properties"]["date"]["value"]).date()
        grid_id = record["properties"]["grid_id"]
        properties = {k: v for k, v in record["properties"].items() if k not in ["date", "grid_id"]}
        # Flatten the properties into a single dictionary
        return {
            "date": date,
            "grid_id": grid_id,
            **properties,
        }

    plan = feature_plan.planned_collection

    features = plan.getInfo()["features"]
    rows = [convert_record(f) for f in features]
    return pl.DataFrame(rows)
