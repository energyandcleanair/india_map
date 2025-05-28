#!/usr/bin/env python3
"""
Run PM₂.₅ Prediction Pipeline

This script performs the following steps:
  1. Loads pre‐engineered PM₂.₅ data from a CSV file.
  2. Drops unnecessary columns and merges with a grid monitor shapefile to obtain region info.
  3. Splits the data by region (using k_region) and creates 10‐fold GroupKFold splits 
     (grouped by grid_id_50km). For each fold, the corresponding training folds across regions 
     are concatenated to form an overall training set (and similarly for test).
  4. Trains an XGBoost model on each fold using fixed parameters, reports performance metrics 
     and extracts feature importances.
  5. Loads a separate dataset for prediction, generates PM₂.₅ predictions (setting negatives to 0),
     and saves the output.

All file paths are constructed using the global variable `path_to_data`.
"""

import os
import sys
import math
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# ------------------------------
# Global path variable (update as needed)
# ------------------------------
path_to_data = "../../input_data/"  # update this as needed

# ------------------------------
# Helper function: Create 10-fold splits for one region
# ------------------------------


def create_region_folds(df_region):
    df_region = df_region.reset_index(drop=True)
    y_region = df_region.pop('pm25').to_frame()
    X_region = df_region.copy()
    gkf = GroupKFold(n_splits=10)
    folds_train, folds_test = [], []
    for train_idx, test_idx in gkf.split(X_region, y_region, groups=X_region['grid_id_50km']):
        folds_train.append(df_region.loc[train_idx])
        folds_test.append(df_region.loc[test_idx])
    return folds_train, folds_test

# ------------------------------
# Main Pipeline
# ------------------------------


def main():
    record = 'xgb'  # identifier for this run

    # Step 1: Load pre-engineered PM2.5 data
    df_path = os.path.join(path_to_data, "df_ml.csv")
    if os.path.exists(df_path):
        pm25 = pd.read_csv(df_path)
    else:
        # read parquet file from path_to_data
        pm25 = pd.read_parquet(os.path.join(
            path_to_data, "df_ml.parquet"))

    # only take a sample for testing (and use another sampler for mocking the
    # data for prediction)
    pm25 = pm25.iloc[::100]

    print("Original data shape:", pm25.shape)
    print("Columns:", list(pm25.columns))
    print("Missing values:\n", pm25.isna().sum())

    # Step 2: Drop unneeded columns (adjust list as needed)
    drop_cols = [
        'omi_no2', 'aot_daily_annual', 'co_daily_annual', 'omi_no2_annual',
        'v_wind_annual', 'u_wind_annual', 'rainfall_annual',
        'thermal_radiation_annual', 'low_veg_annual', 'high_veg_annual',
        'dewpoint_temp_annual', 'wind_degree_annual', 'RH_annual',
        'NO2_tropos_annual', 'aod_annual', 'CO_annual', 'co_daily_allyears',
        'NO2_tropos_allyears', 'aod_allyears', 'cos_day_of_year'
    ]
    pm25 = pm25.drop(columns=drop_cols, errors='ignore')
    print("Data shape after dropping columns:", pm25.shape)

    # Step 3: Merge with grid monitor shapefile to obtain region info
    grid_shp_path = os.path.join(
        path_to_data, "intermediate", "grid_india_monitor_region")
    grid_india = gpd.read_file(grid_shp_path)
    grid_india = grid_india[['grid_id',
                             'grid_id_50km', 'k_region', 'geometry']].copy()
    # Ensure grid_id is string
    pm25['grid_id'] = pm25['grid_id'].astype(int).astype(str)
    grid_india['grid_id'] = grid_india['grid_id'].astype(int).astype(str)
    pm25 = pd.merge(pm25, grid_india, how='left', on='grid_id')
    print("Data shape after merging:", pm25.shape)
    print("Unique k_region values:", pm25['k_region'].unique())

    # Step 4: Create 10-fold splits per region and then concatenate corresponding folds
    region1 = pm25.loc[pm25['k_region'] == 1]
    region2 = pm25.loc[pm25['k_region'] == 2]
    region3 = pm25.loc[pm25['k_region'] == 3]

    train_folds_r1, test_folds_r1 = create_region_folds(region1)
    train_folds_r2, test_folds_r2 = create_region_folds(region2)
    train_folds_r3, test_folds_r3 = create_region_folds(region3)

    train_concat, test_concat = [], []
    for i in range(10):
        fold_train = pd.concat(
            [train_folds_r1[i], train_folds_r2[i], train_folds_r3[i]], axis=0, ignore_index=True)
        fold_train = shuffle(fold_train).reset_index(drop=True)
        train_concat.append(fold_train)
        fold_test = pd.concat(
            [test_folds_r1[i], test_folds_r2[i], test_folds_r3[i]], axis=0, ignore_index=True)
        fold_test = shuffle(fold_test).reset_index(drop=True)
        test_concat.append(fold_test)
    print("Created 10 concatenated train/test folds.")

    # Step 5: Outer CV training using XGBoost
    # Fixed best parameters for XGBoost (adjust as needed)
    best_params_xgb = {
        'eta': 0.01,
        'gamma': 0.8,
        'max_depth': 30,
        'min_child_weight': 0.8,
        'n_estimators': 1500,
        'subsample': 0.8,
        'lambda': 1,
        'booster': 'gbtree'
    }

    trn_r2, trn_rmse, cv_r2, cv_rmse = [], [], [], []
    xgb_feature_importances = []
    train_dfs, eval_dfs = [], []

    # Define the feature list (adjust as needed)
    feature_cols = [
        'aot_daily', 'co_daily', 'v_wind', 'u_wind', 'rainfall', 'temp',
        'pressure', 'thermal_radiation', 'low_veg', 'high_veg', 'dewpoint_temp',
        'month', 'day_of_year', 'monsoon', 'lon', 'lat',
        'wind_degree', 'RH', 'aot_rolling', 'co_rolling', 'omi_no2_rolling',
        'v_wind_rolling', 'u_wind_rolling', 'rainfall_rolling', 'temp_rolling',
        'wind_degree_rolling', 'RH_rolling'
    ]
    # (Adjust feature list if needed)

    for fold in range(10):
        train_fold = train_concat[fold].copy()
        test_fold = test_concat[fold].copy()

        # Save identifying columns for record keeping
        train_df = train_fold[['date', 'grid_id', 'k_region']].copy()
        eval_df = test_fold[['date', 'grid_id', 'k_region']].copy()

        # Separate target and features
        # y_trn = train_fold.pop("pm25").to_frame()
        # ISSUE pop("pm25") is already done in create_region_folds, so column is already
        # removed, and train_fold is already a dataframe
        y_trn = train_fold.copy()
        X_trn = train_fold.copy()
        # ISSUE same as above
        # y_val = test_fold.pop("pm25").to_frame()
        y_val = test_fold.copy()
        X_val = test_fold.copy()

        # Drop columns not used for modeling
        drop_cols = ['date', 'grid_id', 'grid_id_50km', 'k_region', 'geometry']
        X_trn_model = X_trn.drop(columns=drop_cols, errors='ignore')
        X_val_model = X_val.drop(columns=drop_cols, errors='ignore')

        # ISSUE drop geometry and date columns, cause problems with model
        y_trn = y_trn.drop(columns=['geometry', 'date'], errors='ignore')
        y_val = y_val.drop(columns=['geometry', 'date'], errors='ignore')

        # Train XGBoost model
        model_xgb = XGBRegressor(
            **best_params_xgb, n_jobs=int(os.getenv("SLURM_CPUS_PER_TASK", 4)), tree_method="hist")
        model_xgb.fit(X_trn_model, y_trn.values.ravel())

        # Get feature importance (using gain)
        booster = model_xgb.get_booster()
        fi = booster.get_score(importance_type="gain")
        fi_df = pd.DataFrame(data=list(fi.items()), columns=[
                             "feature", "gain"]).sort_values(by="gain", ascending=False)
        xgb_feature_importances.append(fi_df)

        # Compute training metrics
        y_trn_pred = model_xgb.predict(X_trn_model)
        r2_trn = r2_score(y_trn, y_trn_pred)
        rmse_trn = math.sqrt(mean_squared_error(y_trn, y_trn_pred))
        trn_r2.append(r2_trn)
        trn_rmse.append(rmse_trn)
        # ISSUE this crashes due to mismatch in length of y_trn and y_trn_pred
        # train_df['trn_y_pred'] = y_trn_pred
        train_dfs.append(train_df)

        # Compute test metrics
        y_val_pred = model_xgb.predict(X_val_model)
        # ISSUE this crashes due to dimension mismatch
        # eval_df['y_pred'] = y_val_pred
        eval_dfs.append(eval_df)
        r2_val = r2_score(y_val, y_val_pred)
        rmse_val = math.sqrt(mean_squared_error(y_val, y_val_pred))
        cv_r2.append(r2_val)
        cv_rmse.append(rmse_val)

        print(f"Fold {fold+1}: Train R²: {r2_trn:.3f}, Train RMSE: {rmse_trn:.3f} | CV R²: {r2_val:.3f}, CV RMSE: {rmse_val:.3f}")

    print("\nOverall Performance:")
    print(
        f"Average Train R²: {np.mean(trn_r2):.3f}, Average Train RMSE: {np.mean(trn_rmse):.3f}")
    print(
        f"Average CV R²: {np.mean(cv_r2):.3f}, Average CV RMSE: {np.mean(cv_rmse):.3f}")

    # Save feature importance and fold predictions
    output_ml_dir = os.path.join(path_to_data, "intermediate", "ML_full_model")
    for i in range(10):
        fi_path = os.path.join(
            output_ml_dir, f"{record}_fold_{i+1}_feature_importance.csv")
        train_path = os.path.join(
            output_ml_dir, f"{record}_fold_{i+1}_traindf.csv")
        eval_path = os.path.join(
            output_ml_dir, f"{record}_fold_{i+1}_evaldf.csv")
        xgb_feature_importances[i].to_csv(fi_path, index=False)
        train_dfs[i].to_csv(train_path, index=False)
        eval_dfs[i].to_csv(eval_path, index=False)

    # Step 6: Final Prediction on New Data
    pred_file = os.path.join(path_to_data, "intermediate",
                             "ML_full_model", "df_to_be_predicted.csv")
    if os.path.exists(pred_file):
        df_pred = pd.read_csv(pred_file)

    else:
        # mocking data with the same columns as the original data
        df_pred = pd.read_parquet(os.path.join(
            path_to_data, "df_ml.parquet"))

        # drop column pm25
        df_pred = df_pred.drop(columns=['pm25'], errors='ignore')

        df_pred = df_pred.iloc[10::100]
        # drop cols, same way as above
        drop_cols = [
            'omi_no2', 'aot_daily_annual', 'co_daily_annual', 'omi_no2_annual',
            'v_wind_annual', 'u_wind_annual', 'rainfall_annual',
            'thermal_radiation_annual', 'low_veg_annual', 'high_veg_annual',
            'dewpoint_temp_annual', 'wind_degree_annual', 'RH_annual',
            'NO2_tropos_annual', 'aod_annual', 'CO_annual', 'co_daily_allyears',
            'NO2_tropos_allyears', 'aod_allyears', 'cos_day_of_year'
        ]
        df_pred = df_pred.drop(columns=drop_cols, errors='ignore')

    # Prepare features by dropping non-model columns
    X_fin = df_pred.drop(columns=[
                         'date', 'grid_id', 'grid_id_50km', 'k_region', 'geometry'], errors='ignore')
    # Here we use the last fold's model for final prediction (or retrain on full data)
    final_pred = model_xgb.predict(X_fin)
    # ISSUE this does not work because of dimension mismatch
    # df_pred['pm25_pred_xgb'] = final_pred
    # Replace negative predictions with 0
    # ISSUE
    # df_pred.loc[df_pred['pm25_pred_xgb'] < 0, 'pm25_pred_xgb'] = 0
    out_pred_file = os.path.join(path_to_data, "intermediate",
                                 "ML_full_model", f"pm25_pred_{record}_negatives_replaced.csv")
    df_pred.to_csv(out_pred_file, index=False)
    print("Final predictions saved to:", out_pred_file)


if __name__ == '__main__':
    main()
