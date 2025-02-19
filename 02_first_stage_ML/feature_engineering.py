#!/usr/bin/env python3
"""
Feature Engineering for ML Model Update

This script performs the following steps:
  1. Loads and concatenates all CSV files from the "features" directory.
  2. Processes the concatenated data by:
     - Adding time features (date, month, day_of_year, cos_day_of_year, monsoon)
     - Concatenating grid centroid data (grid_id, lon, lat) from a shapefile
     - Computing wind direction and relative humidity
     - Computing 7‑day rolling averages for select variables
     - Creating annual and overall (all-years) aggregate features
  3. Saves the final processed DataFrame as "df_for_imputation.csv"

All file paths are defined relative to the global variable **path_to_data**.
"""

import os
import sys
import math
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from dateutil.relativedelta import relativedelta
from metpy.units import units
import metpy.calc as mpcalc
from metpy.calc import relative_humidity_from_dewpoint

# ------------------------------
# Global path variable (update as needed)
# ------------------------------
path_to_data = "path_to_data"  # Replace with your base data directory

# ------------------------------
# Section 1: Data Loading & Concatenation
# ------------------------------
def load_and_concatenate_csvs(input_dir):
    """
    Load all CSV files from the input directory and concatenate them.
    Assumes that each CSV file has the same columns.
    """
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                 if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in", input_dir)
        sys.exit(1)
    dfs = [pd.read_csv(file) for file in csv_files]
    concat_df = pd.concat(dfs, ignore_index=True)
    print(f"Concatenated DataFrame shape: {concat_df.shape}")
    return concat_df

# ------------------------------
# Section 2: Feature Engineering Functions
# ------------------------------
def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    conditions = [
        df['date'].dt.month.isin([6, 7, 8, 9]),
        df['date'].dt.month.isin([12, 1, 2, 3, 4, 5, 10, 11])
    ]
    choices = [1, 0]
    df['monsoon'] = np.select(conditions, choices, default=-1)
    return df

def concatenate_grid_centroids(df, grid_shp_path):
    """
    Load grid centroid data from the shapefile and concatenate the lon/lat
    columns to the DataFrame based on grid_id.
    (Here we merge the grid centroid DataFrame with the features DataFrame.)
    """
    grid = gpd.read_file(grid_shp_path)
    grid = grid.to_crs(epsg=4326)
    grid['lon'] = grid.centroid.x  
    grid['lat'] = grid.centroid.y
    grid['grid_id'] = grid['grid_id'].astype(int).astype(str)
    grid_df = pd.DataFrame(grid[['grid_id', 'lon', 'lat']])
    df = pd.merge(df, grid_df, on='grid_id', how='left')
    return df

def add_wind_and_rh(df):
    u = df['u_wind'].to_numpy() * units("m/s")
    v = df['v_wind'].to_numpy() * units("m/s")
    df['wind_degree'] = mpcalc.wind_direction(u, v)
    temp_vals = df['temp'].to_numpy() * units.degC
    dew_vals  = df['dewpoint_temp'].to_numpy() * units.degC
    df['RH'] = relative_humidity_from_dewpoint(temp_vals, dew_vals)
    return df

def compute_rolling_averages(df, window=7):
    df = df.sort_values(['grid_id', 'date'])
    cols_to_roll = ['aot_daily', 'co_daily', 'omi_no2', 'v_wind', 'u_wind',
                      'rainfall', 'temp', 'pressure', 'thermal_radiation',
                      'dewpoint_temp', 'wind_degree', 'RH']
    for col in cols_to_roll:
        df[col + '_rolling'] = df.groupby('grid_id')[col]\
                                  .transform(lambda x: x.rolling(window=window, center=True, min_periods=1).mean())
    return df

def add_annual_overall_aggregates(df):
    daily_vars = ['aot_daily', 'co_daily', 'omi_no2', 'v_wind', 'u_wind',
                  'rainfall', 'temp', 'pressure', 'thermal_radiation',
                  'dewpoint_temp', 'wind_degree', 'RH']
    annual = df[['date', 'grid_id'] + daily_vars].copy()
    annual['date'] = pd.to_datetime(annual['date'])
    annual_avg = annual.groupby(['grid_id', pd.Grouper(key='date', freq='Y')]).mean().reset_index()
    annual_avg['year'] = annual_avg['date'].dt.year.astype(str)
    whole_avg = annual.groupby('grid_id').mean().reset_index()
    df['year'] = df['date'].dt.year.astype(str)
    df = df.merge(annual_avg, on=['grid_id', 'year'], suffixes=("", "_annual"))
    whole_avg['grid_id'] = whole_avg['grid_id'].astype(str)
    df = df.merge(whole_avg, on='grid_id', suffixes=("", "_allyears"))
    return df

# ------------------------------
# Section 3: Main Feature Engineering Pipeline
# ------------------------------
def main():
    # Define the directory where feature CSVs are stored
    input_dir = os.path.join(path_to_data, "features")
    df_concat = load_and_concatenate_csvs(input_dir)
    
    # Apply feature engineering steps
    df_processed = add_time_features(df_concat)
    grid_shp = os.path.join(path_to_data, "grid_india_10km.shp")
    df_processed = concatenate_grid_centroids(df_processed, grid_shp)
    df_processed = add_wind_and_rh(df_processed)
    df_processed = compute_rolling_averages(df_processed, window=7)
    df_processed = add_annual_overall_aggregates(df_processed)
    
    # Save the processed DataFrame for model input
    output_file = os.path.join(path_to_data, "df_for_imputation.csv")
    df_processed.drop_duplicates(inplace=True)
    print(f"Processed DataFrame shape: {df_processed.shape}")
    print("Missing values:\n", df_processed.isna().sum())
    print(f"Date range: {df_processed['date'].min()} to {df_processed['date'].max()}")
    df_processed.to_csv(output_file, index=False)
    print("Processed features saved to:", output_file)

if __name__ == '__main__':
    main()
