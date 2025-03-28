#!/usr/bin/env python3
"""
Run NO₂ Model Update Pipeline

This script performs the following steps:
  1. Loads the pre‐processed (feature‐engineered) NO₂ data from a CSV file.
  2. Loads the sampled training data (already feature‐engineered) for the NO₂ model.
  3. Trains a NO₂ prediction model using inner and outer cross‐validation.
  4. Performs final imputation on rows with missing NO₂ values using the trained model.

All file paths are defined by the variable `path_to_data`.

Adjust parameters and settings as needed.
"""

import os
import sys
import math
import random
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# ------------------------------
# Global path variable (update as needed)
# ------------------------------
path_to_data = "path_to_data"  # Use this for all file I/O

# ------------------------------
# Section: ML Model Training Functions
# ------------------------------


def run_inner_cv(X_inner, y_inner, groups):
    """
    Runs inner CV (using GroupKFold) to tune XGBoost hyperparameters.
    Returns the best parameters and the inner CV splits.
    """
    gkf_inner = GroupKFold(n_splits=5)
    inner_cv = list(gkf_inner.split(X_inner, y_inner, groups=groups))
    pipe_XGB = XGBRegressor(tree_method='gpu_hist')  # use GPU if available
    params_XGB = {
        'eta': [0.01, 0.1],
        'gamma': [0.8],
        'max_depth': [20, 30],
        'min_child_weight': [0.8, 1],
        'lambda': [10, 100],
        'n_estimators': [1000, 1500],
        'booster': ['gbtree']
    }
    scoring = {'r_squared': 'r2', 'rmse': 'neg_root_mean_squared_error'}
    search = GridSearchCV(
        estimator=pipe_XGB,
        param_grid=params_XGB,
        n_jobs=int(os.getenv("SLURM_CPUS_PER_TASK", 4)),
        scoring=scoring,
        refit='rmse',
        cv=inner_cv,
        verbose=1,
        return_train_score=True
    )
    search.fit(X_inner, y_inner.values.ravel())
    best_params = search.best_params_
    print("Best XGB Parameters (inner CV):", best_params)
    return best_params, inner_cv


def run_outer_cv(df, features, target, group_col, best_params, output_dir):
    """
    Runs outer CV (using GroupKFold) with the provided best parameters.
    Saves fold-level results (feature importance, training and evaluation data).
    Returns the model from the last outer fold.
    """
    X = df[features].copy()
    y = df[target].copy()
    gkf_outer = GroupKFold(n_splits=10)
    outer_cv = list(gkf_outer.split(X, y, groups=df[group_col]))

    trn_r2, trn_rmse, cv_r2, cv_rmse = [], [], [], []
    dfs_importance, train_dfs, eval_dfs = [], [], []
    drop_cols = ['date', 'grid_id', 'grid_id_50km', 'year_month']

    for fold, (trn_idx, val_idx) in enumerate(outer_cv):
        print(f"\n===== Outer Fold {fold+1} =====")
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        train_df = X_trn[['date', 'grid_id']].copy()
        train_df['y_trn'] = y_trn
        eval_df = X_val[['date', 'grid_id']].copy()
        eval_df['y_val'] = y_val

        X_trn_model = X_trn.drop(columns=drop_cols, errors='ignore')
        X_val_model = X_val.drop(columns=drop_cols, errors='ignore')

        model_xgb = XGBRegressor(
            **best_params, n_jobs=int(os.getenv("SLURM_CPUS_PER_TASK", 4)), tree_method='gpu_hist')
        model_xgb.fit(X_trn_model, y_trn.values.ravel())

        # Record feature importances
        imp_df = pd.DataFrame({
            'feature': X_trn_model.columns,
            'importance': model_xgb.feature_importances_
        }).sort_values(by='importance', ascending=False)
        dfs_importance.append(imp_df)

        # Training performance
        y_trn_pred = model_xgb.predict(X_trn_model)
        r2_trn = r2_score(y_trn, y_trn_pred)
        rmse_trn = math.sqrt(mean_squared_error(y_trn, y_trn_pred))
        trn_r2.append(r2_trn)
        trn_rmse.append(rmse_trn)
        train_df['trn_y_pred'] = y_trn_pred
        train_dfs.append(train_df)

        # Validation performance
        y_val_pred = model_xgb.predict(X_val_model)
        eval_df['y_pred'] = y_val_pred
        eval_dfs.append(eval_df)
        r2_val = r2_score(y_val, y_val_pred)
        rmse_val = math.sqrt(mean_squared_error(y_val, y_val_pred))
        cv_r2.append(r2_val)
        cv_rmse.append(rmse_val)

        print(f"Fold {fold+1}: Train R2: {r2_trn:.3f}, Train RMSE: {rmse_trn:.3f} | CV R2: {r2_val:.3f}, CV RMSE: {rmse_val:.3f}")

    print("\nOverall Performance:")
    print(
        f"Average Train R2: {np.mean(trn_r2):.3f}, Average Train RMSE: {np.mean(trn_rmse):.3f}")
    print(
        f"Average CV R2: {np.mean(cv_r2):.3f}, Average CV RMSE: {np.mean(cv_rmse):.3f}")

    for i in range(len(outer_cv)):
        dfs_importance[i].to_csv(os.path.join(
            output_dir, f"fold_{i+1}_XGB_feature_importance.csv"), index=False)
        train_dfs[i].to_csv(os.path.join(
            output_dir, f"fold_{i+1}_XGB_traindf.csv"), index=False)
        eval_dfs[i].to_csv(os.path.join(
            output_dir, f"fold_{i+1}_XGB_evaldf.csv"), index=False)

    return model_xgb

# ------------------------------
# Main Pipeline
# ------------------------------


def main():
    # Note: Feature engineering is assumed to have been completed separately.
    # The pre-processed data are saved in "df_for_imputation.csv" and the sampled
    # training data in "NO2_ml_df_sampled.csv" within the ML_full_model folder.

    # Step 1: (Optional) Verify pre-processed features by loading df_for_imputation.csv
    fe_file = os.path.join(path_to_data, "df_for_imputation.csv")
    if os.path.exists(fe_file):
        df_fe = pd.read_csv(fe_file)
        print(
            f"Feature-engineered data loaded from {fe_file} with shape: {df_fe.shape}")
    else:
        print(
            f"Feature-engineered file {fe_file} not found. Please run the feature engineering script first.")
        sys.exit(1)

    # Step 2: Load the sampled training data for the NO₂ model
    ml_input_file = os.path.join(
        path_to_data, "ML_full_model", "NO2_ml_df_sampled.csv")
    df_ml = pd.read_csv(ml_input_file)
    df_ml['date'] = pd.to_datetime(df_ml['date'])
    df_ml['grid_id'] = df_ml['grid_id'].astype(str)

    # Define features and target (adjust as needed)
    feature_cols = ['aot_daily', 'co_daily', 'v_wind', 'u_wind', 'rainfall', 'temp',
                    'pressure', 'thermal_radiation', 'low_veg', 'high_veg', 'dewpoint_temp',
                    'month', 'day_of_year', 'cos_day_of_year', 'monsoon', 'lon', 'lat',
                    'wind_degree', 'RH', 'aot_rolling', 'co_rolling', 'omi_no2_rolling',
                    'v_wind_rolling', 'u_wind_rolling', 'rainfall_rolling', 'temp_rolling',
                    'wind_degree_rolling', 'RH_rolling', 'thermal_radiation_rolling',
                    'dewpoint_temp_rolling', 'aot_daily_annual', 'co_daily_annual',
                    'omi_no2_annual', 'v_wind_annual', 'u_wind_annual', 'rainfall_annual',
                    'thermal_radiation_annual', 'low_veg_annual', 'high_veg_annual',
                    'dewpoint_temp_annual', 'wind_degree_annual', 'RH_annual', 'co_daily_allyears']
    target_col = 'NO2_tropos'

    # Step 3: (Optional) Run inner CV tuning to get best XGBoost parameters
    X_inner = df_ml.drop(
        columns=['date', 'grid_id', 'grid_id_50km', 'year_month'], errors='ignore')
    y_inner = df_ml[target_col]
    groups_inner = df_ml['grid_id_50km']  # Assumes this column exists
    best_params_XGB, inner_cv = run_inner_cv(X_inner, y_inner, groups_inner)

    # Step 4: Run outer CV training with the tuned parameters
    OUTPUT_ML_DIR = os.path.join(path_to_data, "NO2_impute")
    final_model = run_outer_cv(df_ml, feature_cols, target_col, 'grid_id_50km',
                               best_params_XGB, output_dir=OUTPUT_ML_DIR)

    # Step 5: Final Imputation on Missing NO₂ Data
    missing_file = os.path.join(path_to_data, "NO2_missing_to_be_imputed.csv")
    df_missing = pd.read_csv(missing_file)
    df_missing['date'] = pd.to_datetime(df_missing['date'])
    X_missing = df_missing[feature_cols].copy()
    pred_missing = final_model.predict(X_missing)
    df_missing['NO2_imputed'] = pred_missing
    imputed_out = os.path.join(OUTPUT_ML_DIR, "NO2_imputed_XGB.csv")
    df_missing.to_csv(imputed_out, index=False)
    print("Final imputed NO₂ saved to:", imputed_out)


if __name__ == '__main__':
    main()
