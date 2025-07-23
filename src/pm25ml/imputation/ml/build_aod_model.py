"""Builds an XGBoost AOD imputation model, and evaluates its performance."""

import math
import os
import tempfile
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd
import polars as pl
from arrow import Arrow
from google.cloud.storage import Client as GCSClient
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.utils import shuffle
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

MAX_PARALLEL_TASKS = int(os.getenv("MAX_PARALLEL_TASKS", "8"))

MODEL_STORAGE_BUCKET = "crea-pm25ml-models"


def main(extra_sampler: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x) -> None:
    """Run imputation ML model for AOD."""
    # 1. Sampling
    logger.info("Loading and sampling training data for AOD imputation")
    df_sampled = extra_sampler(load_training_data())

    # 2. Create folds
    # outer_cv is a list where each item contains a tuple with indices
    # of training and validation sets for each fold
    logger.info("Creating outer cross-validation folds")
    outer_cv = make_folds(df_sampled, n_folds=10)

    # 3. Finding optimal hyperparameters (inner CV)
    logger.info("Finding optimal hyperparameters (inner CV)")
    best_params_xgb, hyper_tuning_metrics = find_hyper_params(
        df_sampled,
        outer_cv[0],
    )

    # cross checked, these best parameters from the code are the same as in the paper

    # 4. Training imputation model (fit XGBRegressor, compute training metrics)
    logger.info("Training imputation model with XGBRegressor")
    model, training_diagnostics = train_model(
        df_sampled,
        outer_cv,
        best_params_xgb,
    )

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


def make_folds(df_sampled: pd.DataFrame, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Make the folds for inner and outer cross-validation.

    Cross-validation is used to ensure that the model is trained and evaluated
    on different data, for intro see https://scikit-learn.org/stable/modules/cross_validation.html
    Here, df_sampled should already be the training data set, with the test set
    being set apart before. This function creates the cross-validation folds
    and returns the indexes for the training and validation sets for each fold.

    Using GroupKFold to ensure that the same grid__id_50km is not in both
    training and testing sets.

    https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold.split

    """
    target = df_sampled[[AOD_COLUMN]]
    predictors = df_sampled.drop(columns=[AOD_COLUMN])

    # NOTE because using the default shuffle=False, not possible to give the random_state
    # is the result reproducible?
    gkf = GroupKFold(n_splits=n_folds)

    # split returns the indices of the training and testing sets
    outer_cv = gkf.split(predictors, target, groups=predictors["grid__id_50km"])
    return list(outer_cv)


def find_hyper_params(
    df_sampled: pd.DataFrame,
    selected_outer_cv: tuple[np.ndarray, np.ndarray],
) -> tuple[dict, dict]:
    """
    Find the optimal hyperparameters for the model.

    This function performs hyperparameter tuning using RandomizedSearchCV
    on the XGBRegressor model. It uses data from 1 one of the outer
    cross validation folds.

    The paper states that for AOD imputation, the XGBoost model was used.
    Therefore ingoring the part with LightGBM in the original code.

    Args:
        df_sampled (pd.DataFrame): Dataframe with sampled data for training.
        selected_outer_cv (tuple): A tuple containing the indices of the
            training and validation sets for the selected outer fold.
        tree_method (str): The tree method to use for XGBRegressor.

    """
    # Get the training set for the selected outer fold
    # selected_outer_cv is a tuple of arrays, where the first array contains
    # the indices for the training set and the second array contains the indices
    # for the validation set for the selected outer fold
    inner_train = df_sampled.loc[selected_outer_cv[0]]

    # Shuffle the training set to ensure randomness. Could use a random_state
    # for reproducibility, but this not done in the original code.
    inner_train = cast("pd.DataFrame", shuffle(inner_train)).reset_index(drop=True)

    # Separate the target variable 'aod' from the features
    target_inner = inner_train[[AOD_COLUMN]]
    predictors_inner = inner_train.drop(columns=[AOD_COLUMN])

    # Create the inner cross-validation folds using GroupKFold
    gkf = GroupKFold(n_splits=5)
    inner_cv = gkf.split(predictors_inner, target_inner, groups=predictors_inner["grid__id_50km"])

    n_parallel_searches = 3

    # Use RandomizedSearchCV to tune hyperparameters for xgbRegressor
    pipe_xgb = XGBRegressor(
        tree_method="hist",
        n_jobs=MAX_PARALLEL_TASKS / n_parallel_searches,
    )

    params_xgb = {
        "eta": [0.1],
        "gamma": [0.8],
        "max_depth": [20],
        "min_child_weight": [1],
        "subsample": [0.8],
        "lambda": [100],
        "n_estimators": [1000],
        "booster": ["gbtree"],
    }

    xgb_search = RandomizedSearchCV(
        pipe_xgb,
        params_xgb,
        n_jobs=n_parallel_searches,
        scoring={
            "r_squared": "r2",
            "rmse": "neg_root_mean_squared_error",
        },
        refit="rmse",
        cv=list(inner_cv),
        verbose=1,
        return_train_score=True,
    )

    xgb_search.fit(
        predictors_inner.drop(columns=INDEX_COLUMNS),
        target_inner.to_numpy().ravel(),
    )

    # Get the best hyperparameters
    best_params_xgb = xgb_search.best_params_
    logger.info(f"XGB best parameters: {best_params_xgb}")

    # Get the performance metrics for the best hyperparameters
    results = xgb_search.cv_results_

    # find the index of the best parameters in the results
    for i, x in enumerate(results["params"]):
        if x == best_params_xgb:
            index = i
            break
    else:
        msg = "best_params_xgb not found in results"
        raise ValueError(msg)

    val_r2 = results["mean_test_r_squared"][index]
    train_r2 = results["mean_train_r_squared"][index]
    val_rmse = results["mean_test_rmse"][index]
    train_rmse = results["mean_train_rmse"][index]

    # Return the best hyperparameters and performance metrics
    return best_params_xgb, {
        "val_r2": val_r2,
        "train_r2": train_r2,
        "val_rmse": val_rmse,
        "train_rmse": train_rmse,
    }


def train_model(
    df_sampled: pd.DataFrame,
    outer_cv: list[tuple[np.ndarray, np.ndarray]],
    best_params_xgb: dict,
) -> tuple[XGBRegressor, dict]:
    """
    Train the imputation model using XGBRegressor.

    This function trains the XGBRegressor model using the sampled data
    and the outer cross-validation folds. It computes the training and
    validation scores and extracts feature importances for each fold.
    Unlike in the original code, the full predicted test and validation
    data are not stored (variable train_dfs and eval_dfs in original code).

    Args:
        df_sampled (pd.DataFrame): Dataframe with sampled data for training.
        outer_cv (list): List of tuples with indices for training and testing sets.
        best_params_xgb (dict): Dictionary with hyperparameters for XGBRegressor.
        tree_method (str): Method for tree construction in XGBRegressor, default
            is 'gpu_hist' (following the original code). If no GPU available, use 'hist'.

    """
    # For training and testing metrics, create dataframes to store the results
    trn_metrics = pd.DataFrame(
        index=range(len(outer_cv)),
        columns=["train_r2", "train_rmse", "cv_r2", "cv_rmse"],
    )
    # To collect feature importance from each fold, create list of dataframes
    # (to be merged at the end, more efficient than appending to a df in each iteration)
    df_feat_imp = []

    # Generate the target variable and features
    # y is the column to be predicted, X is the rest of the data
    target = df_sampled[[AOD_COLUMN]]
    predictors = df_sampled.drop(columns=[AOD_COLUMN, *INDEX_COLUMNS])

    if len(outer_cv) == 0:
        msg = "No outer cross-validation folds found. Check the input data."
        raise ValueError(msg)

    # Initialize model_xgb to ensure it is always defined
    model_xgb = None

    # Loop through the outer cross-validation folds
    # For each fold, train the model and evaluate it on the validation set
    for n_fold, (trn_idx, val_idx) in enumerate(outer_cv):
        logger.info(f"Training fold {n_fold + 1} of {len(outer_cv)}")
        predictors_trn, predictors_val = predictors.iloc[trn_idx], predictors.iloc[val_idx]
        target_trn, target_val = target.iloc[trn_idx], target.iloc[val_idx]

        # Train the model using given hyperparameters
        # Note, that it would be possible to give evaluation metric(s) here, but in
        # then the metric would also be used for early stopping, which we don't want in this case.
        model_xgb = XGBRegressor(**best_params_xgb, n_jobs=MAX_PARALLEL_TASKS, tree_method="hist")
        model_xgb.fit(predictors_trn, target_trn.to_numpy().ravel())

        # Get the importances of features (for logging and analysis)
        importance_df = pd.DataFrame(
            {
                "feature": predictors_trn.columns,
                "importance": model_xgb.feature_importances_,
                "fold": n_fold,
            },
        ).sort_values(by=["importance"], ascending=False)
        df_feat_imp.append(importance_df)

        # Predict on the training set and compute training metrics
        trn_y_pred = model_xgb.predict(predictors_trn)

        trn_metrics.loc[n_fold, "train_r2"] = r2_score(target_trn, trn_y_pred)
        trn_metrics.loc[n_fold, "train_rmse"] = math.sqrt(
            mean_squared_error(target_trn, trn_y_pred),
        )

        # Predict on the validation set and compute validation metrics
        y_pred = model_xgb.predict(predictors_val.values)

        trn_metrics.loc[n_fold, "cv_r2"] = r2_score(target_val, y_pred)
        trn_metrics.loc[n_fold, "cv_rmse"] = math.sqrt(mean_squared_error(target_val, y_pred))

    if model_xgb is None:
        msg = "Model is not trained yet. Check the input data and hyperparameters."
        raise ValueError(msg)

    # The final score of the cross validated model are the means of the scores
    # from all folds.

    train_metrics_summary = trn_metrics.aggregate(
        {
            "train_r2": ["mean", "std", "min", "max"],
            "train_rmse": ["mean", "std", "min", "max"],
            "cv_r2": ["mean", "std", "min", "max"],
            "cv_rmse": ["mean", "std", "min", "max"],
        },
    )

    # Diagnostics output: feature importances
    # (merge lists of df's to a single df)
    combined_feat_imp = pd.concat(df_feat_imp, ignore_index=True)

    logger.info(f"Training metrics:\n {trn_metrics}")
    logger.info(f"Training metrics summary:\n{train_metrics_summary}")
    logger.info(f"Feature importances:\n{combined_feat_imp}")

    return model_xgb, {
        "train_metrics_summary": train_metrics_summary,
        "trn_metrics": trn_metrics,
        "feature_importance": combined_feat_imp,
    }


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
    if os.environ.get("TINY_SAMPLE") == "true":

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
