"""Builds an XGBoost AOD imputation model, and evaluates its performance."""

import math
import os
import tempfile
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl
from arrow import Arrow
from google.cloud.storage import Client as GCSClient
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from xgboost import XGBRegressor

from pm25ml.logging import logger

AOD_INPUT_COLS = [
    "grid_id",
    "grid__id_50km",
    "date",
    "merra_aot__aot",
    "merra_co_top__co",
    "era5_land__v_component_of_wind_10m",
    "era5_land__u_component_of_wind_10m",
    "era5_land__total_precipitation_sum",
    "era5_land__temperature_2m",
    "era5_land__surface_pressure",
    "era5_land__surface_net_thermal_radiation_sum",
    "era5_land__leaf_area_index_low_vegetation",
    "era5_land__leaf_area_index_high_vegetation",
    "era5_land__dewpoint_temperature_2m",
    "modis_aod__Optical_Depth_055",
    "srtm_elevation__elevation",
    "modis_land_cover__water",
    "modis_land_cover__shrub",
    "modis_land_cover__urban",
    "modis_land_cover__forest",
    "modis_land_cover__savanna",
    "month_of_year",
    "day_of_year",
    "cos_day_of_year",
    "monsoon_season",
    "grid__lon",
    "grid__lat",
    "era5_land__c_wind_degree_computed",
    "era5_land__relative_humidity_computed",
    "merra_aot__aot__mean_r7d",
    "merra_co_top__co__mean_r7d",
    "omi_no2__no2__mean_r7d",
    "era5_land__v_component_of_wind_10m__mean_r7d",
    "era5_land__u_component_of_wind_10m__mean_r7d",
    "era5_land__total_precipitation_sum__mean_r7d",
    "era5_land__temperature_2m__mean_r7d",
    "era5_land__c_wind_degree_computed__mean_r7d",
    "era5_land__relative_humidity_computed__mean_r7d",
    "era5_land__surface_net_thermal_radiation_sum__mean_r7d",
    "era5_land__dewpoint_temperature_2m__mean_r7d",
    "merra_aot__aot__mean_year",
    "merra_co_top__co__mean_year",
    "omi_no2__no2__mean_year",
    "era5_land__v_component_of_wind_10m__mean_year",
    "era5_land__u_component_of_wind_10m__mean_year",
    "era5_land__total_precipitation_sum__mean_year",
    "era5_land__surface_net_thermal_radiation_sum__mean_year",
    "era5_land__leaf_area_index_low_vegetation__mean_year",
    "era5_land__leaf_area_index_high_vegetation__mean_year",
    "era5_land__dewpoint_temperature_2m__mean_year",
    "era5_land__c_wind_degree_computed__mean_year",
    "era5_land__relative_humidity_computed__mean_year",
    "merra_co_top__co__mean_all",
]

INDEX_COLUMNS = [
    "grid_id",
    "grid__id_50km",
    "date",
]

AOD_COLUMN = "modis_aod__Optical_Depth_055"

CACHE_DIR = Path(".cache")

MAX_PARALLEL_TASKS = int(os.getenv("MAX_PARALLEL_TASKS", str(os.cpu_count() or 1)))

MODEL_STORAGE_BUCKET = "crea-pm25ml-models"

USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"


def main(extra_sampler: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x) -> None:
    """Run imputation ML model for AOD."""
    # 1. Sampling
    logger.info("Loading and sampling training data for AOD imputation")
    df_sampled = extra_sampler(load_training_data())

    # 2. Create folds
    # outer_cv is a list where each item contains a tuple with indices
    # of training and validation sets for each fold
    logger.info("Creating outer cross-validation folds")
    model = cross_validate_with_stratification(df_sampled)

    del df_sampled

    logger.info("Loading test data for evaluation")
    df_test = extra_sampler(load_test_data())

    # 5. Evaluate model on the test data
    logger.info("Evaluating model on the test data")
    test_metrics = evaluate_model(model, df_test)

    logger.info(f"Test metrics: {test_metrics}")

    # 6. Save the model and diagnostics
    # Create a temporary directory to save diagnostics
    logger.info("Saving model and diagnostics")
    write_model_to_gcs(model)


def cross_validate_with_stratification(df_sampled: pd.DataFrame) -> XGBRegressor:
    """
    Perform cross-validation with stratification on the sampled data.

    Args:
        df_sampled (pd.DataFrame): Sampled data for training.

    Returns:
        XGBRegressor: The trained XGBRegressor model.

    """
    target = df_sampled[[AOD_COLUMN]]
    predictors = df_sampled.drop(columns=[AOD_COLUMN, "grid_id", "date", "grid__id_50km"])
    grouper = df_sampled["grid__id_50km"]

    params_xgb = {
        "eta": 0.1,
        "gamma": 0.8,
        "max_depth": 20,
        "min_child_weight": 1,
        "subsample": 0.8,
        "lambda": 100,
        "n_estimators": 1000,
        "booster": "gbtree",
    }

    n_splits = 10
    cpus_per_model = int(MAX_PARALLEL_TASKS / n_splits)

    model = XGBRegressor(
        **params_xgb,
        n_jobs=cpus_per_model,
        tree_method="hist" if not USE_GPU else "gpu_hist",
    )

    selector = GroupKFold(n_splits=n_splits)

    scores = cross_validate(
        model,
        predictors,
        target,
        cv=selector,
        groups=grouper,
        scoring=["neg_root_mean_squared_error", "r2"],
        n_jobs=n_splits,
        return_train_score=True,
    )

    scores_as_df = pd.DataFrame(scores)
    logger.info(f"Cross-validation scores:\n{scores_as_df}")
    scores_agg = scores_as_df.aggregate(["mean", "std", "min", "max"])
    logger.info(f"Cross-validation scores aggregated:\n{scores_agg}")
    return model


def load_training_data() -> pd.DataFrame:
    """Load the sampled data for AOD imputation from GCS."""
    cache_file = "aod_sampled.parquet"

    if Path(CACHE_DIR, cache_file).exists():
        return pd.read_parquet(Path(CACHE_DIR, cache_file))

    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    results = pl.scan_parquet(
        "gs://crea-pm25ml-combined/stage=sampled/model=aod/",
    )

    to_cache = (
        results.filter(pl.col("split") == "training")
        .with_columns(
            month_of_year=pl.col("date").dt.month(),
        )
        .select(
            [*AOD_INPUT_COLS, "split"],
        )
        .drop("split")
        .collect(engine="streaming")
        .to_pandas()
    )

    to_cache.to_parquet(Path(CACHE_DIR, cache_file))

    return to_cache


def load_test_data() -> pd.DataFrame:
    """Load the test data for AOD imputation from GCS."""
    cache_file = "aod_test.parquet"

    if Path(CACHE_DIR, cache_file).exists():
        return pd.read_parquet(Path(CACHE_DIR, cache_file))

    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    results = pl.scan_parquet(
        "gs://crea-pm25ml-combined/stage=sampled/model=aod/",
    )

    to_cache = (
        results.filter(pl.col("split") == "test")
        .with_columns(
            month_of_year=pl.col("date").dt.month(),
        )
        .select(
            [*AOD_INPUT_COLS, "split"],
        )
        .drop("split")
        .collect(engine="streaming")
        .to_pandas()
    )

    to_cache.to_parquet(Path(CACHE_DIR, cache_file))

    return to_cache


def evaluate_model(model: XGBRegressor, df_rest: pd.DataFrame) -> dict:
    """
    Evaluate the model on the rest of the data.

    This function uses the trained model to predict the AOD values
    for the rest of the data (test set not used for training) and returns
    the predictions.

    Args:
        model (XGBRegressor): The trained XGBRegressor model.
        df_rest (pd.DataFrame): Dataframe with the rest of the data for evaluation.

    """
    # # ~ ~ ~ Original code ~ ~ ~
    # Check if the model is trained
    if not hasattr(model, "feature_importances_"):
        msg = "Model is not trained yet."
        raise ValueError(msg)

    # Predict AOD values
    pred = model.predict(df_rest.drop(columns=[AOD_COLUMN, *INDEX_COLUMNS]))

    # Check if prediction length matches the number of rows in rest_df
    if len(pred) != df_rest.shape[0]:
        msg = "Prediction length does not match the number of rows in rest_df"
        raise ValueError(msg)

    # Calculate metrics for evaluation
    r2 = r2_score(df_rest[AOD_COLUMN], pred)
    rmse = math.sqrt(mean_squared_error(df_rest[AOD_COLUMN], pred))

    return {"r2": r2, "rmse": rmse}


def write_model_to_gcs(model: XGBRegressor) -> None:
    """Save the trained model to Google Cloud Storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir, "aod_imputation_model.json")
        model.save_model(save_path)

        client = GCSClient()
        bucket = client.bucket(MODEL_STORAGE_BUCKET)
        datetime = Arrow.now().isoformat()
        blob = bucket.blob(f"aod_imputation_model_{datetime}.json")
        blob.upload_from_filename(save_path)


if __name__ == "__main__":
    if os.environ.get("TINY_SAMPLE", "false").lower() == "true":

        def _sampler(x: pd.DataFrame) -> pd.DataFrame:
            return (
                x[
                    (x["date"] >= pd.Timestamp("2023-01-01"))
                    & (x["date"] < pd.Timestamp("2023-02-01"))
                ]
                .iloc[::100, :]
                .reset_index(drop=True)
            )
    else:

        def _sampler(x: pd.DataFrame) -> pd.DataFrame:
            return x

    main(
        extra_sampler=_sampler,
    )
