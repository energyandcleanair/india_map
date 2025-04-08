import itertools
import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# This script generates synthetic datasets for a specified range of years and grid IDs.
# Used for testing and benchmarking purposes.

NUMBER_OF_GRIDS = 30_000
YEAR_START = 2015
YEAR_END = 2025

DATASETS = {
  "imputed_no2": [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
  ],
  "imputed_co2": [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
  ],
  "imputed_aod": [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
  ],
  "generated_date": [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
  ],
  "generated_weather": [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
  ],
  "generated_nasa_earthdata": [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
    "feature_7",
  ],
  "nasa_earthdata": [
    "feature_1",
    "feature_2",
    "feature_3",
  ],
  "gee_era5": [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
    "feature_7",
  ],
  "gee_modis": [
    "feature_1",
  ],
  "gee_usgs": [
    "feature_1",
    "feature_2",
    "feature_3",
  ],
  "crea_pm25": [
    "feature_1",
  ],
}

def generate_dataset(
  df: pd.DataFrame,
  features: list[str]
):
  """
  Generates a dataset with the specified features.
  The dataset contains data from 2015 to 2025 for 30,000 grids.
  """
  
  # Generate random data for each feature
  for feature in features:
    print(f"Generating data for {feature}")
    df[feature] = np.random.rand(len(df))
    df[feature] = df[feature].astype(np.double)
  
  return df

def save_dataset(df: pd.DataFrame, filename: str):
  """
  Saves the dataset to a CSV file.
  """
  print(f"Saving dataset to {filename}")
  table = pa.Table.from_pandas(df)

  parquet_format = ds.ParquetFileFormat()

  file_options = ds.ParquetFileFormat().make_write_options(compression='snappy')

  ds.write_dataset(
    table,
    filename,
    format=parquet_format,
    partitioning=ds.partitioning(
      flavor="hive",
      schema=pa.schema([
        ("grid_id", pa.string()),
      ])
    ),
    existing_data_behavior="overwrite_or_ignore",
    file_options=file_options
  )

def generate_base_df():
  
  print("Generating date range")
  # Create a date range for the years and months
  date_range = pd.date_range(start=f"{YEAR_START}-01-01", end=f"{YEAR_END}-12-31", freq="D")
  
  print("Generating grid IDs")
  # Create a grid of grid IDs
  grid_ids = [f"grid_{i}" for i in range(NUMBER_OF_GRIDS)]
  
  print("Generating date*grid_id combinations")
  # Create a DataFrame with all combinations of date and grid ID
  base_df = pd.DataFrame({
      "date": np.repeat(date_range.values, len(grid_ids)),
      "grid_id": np.tile(grid_ids, len(date_range))
  })
  # Ensure correct types for date and grid_id
  base_df['date'] = pd.to_datetime(base_df['date'])
  base_df['grid_id'] = base_df['grid_id'].astype('category')

  base_df.sort_values(by=["grid_id", "date"], inplace=True)

  return base_df

def main():
  base_df = generate_base_df()

  # Generate and save datasets
  for dataset_name, features in DATASETS.items():
    print(f"Generating dataset: {dataset_name}")
    renamed_features = [f"{dataset_name}_{feature}" for feature in features]
    df = generate_dataset(base_df.copy(), renamed_features)
    save_dataset(df, f"data/datasets/{dataset_name}")
    print(f"Dataset {dataset_name} generated and saved.")
    
    del df
    # Free up memory
    import gc
    print("Freeing up memory")
    gc.collect()

if __name__ == "__main__":
  main()