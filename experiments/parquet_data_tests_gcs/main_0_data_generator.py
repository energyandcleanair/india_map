import itertools
import pandas as pd
import numpy as np
import gcsfs

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc

from tqdm import tqdm

from itertools import product

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


def generate_dataset(df: pd.DataFrame, features: list[str]):
    """
    Generates a dataset with the specified features.
    The dataset contains data from 2015 to 2025 for 30,000 grids.
    """

    # Generate random data for each feature
    for feature in features:
        print(f"Generating data for {feature}")
        df[feature] = np.random.rand(len(df))
        df[feature] = df[feature].astype(np.double)

    return pa.Table.from_pandas(df)


def save_dataset(table: pa.Table, filename: str):
    """
    Saves the dataset to a Parquet file on GCS.
    """
    print(f"Saving dataset to {filename}")

    parquet_format = ds.ParquetFileFormat()

    file_options = ds.ParquetFileFormat().make_write_options(compression="snappy")

    base_fs = pa.fs.GcsFileSystem()
    fs = pa.fs.SubTreeFileSystem(base_path=filename, base_fs=base_fs)

    col_index = table.schema.get_field_index("grid_id")
    unique_ids = table.column(col_index).unique()

    partition_col = "grid_id"

    def write_partition(value):
        mask = pc.equal(table[partition_col], value)
        filtered = table.filter(mask)

        ds.write_dataset(
            data=filtered,
            base_dir="",
            filesystem=fs,
            format="parquet",
            partitioning=ds.partitioninsg(
                pa.schema([(partition_col, pa.string())]), flavor="hive"
            ),
            existing_data_behavior="overwrite_or_ignore",
        )
        return value.as_py()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(write_partition, val) for val in unique_ids]

        for future in as_completed(futures):
            print(f"✅ Finished partition: {future.result()}")


def generate_base_df():
    print("Generating date range")
    # Create a date range for the years and months
    dates = pd.date_range(
        start=f"{YEAR_START}-01-01", end=f"{YEAR_END}-12-31", freq="D"
    )

    print("Generating grid IDs")
    # Create a grid of grid IDs
    grid_ids = pd.Series(
        [f"grid_{i}" for i in range(NUMBER_OF_GRIDS)], dtype="category"
    )

    grid_ids_cat = pd.Categorical(grid_ids)

    # Storage
    dfs = []
    chunk_size = 1000

    base_df = pd.DataFrame(columns=["date", "grid_id"])

    for i in tqdm(range(0, len(grid_ids_cat), chunk_size)):
        grid_chunk = grid_ids_cat[i : i + chunk_size]

        date_idx = np.repeat(np.arange(len(dates)), len(grid_chunk))
        grid_idx = np.tile(np.arange(len(grid_chunk)), len(dates))

        df_chunk = pd.DataFrame(
            {"date": dates[date_idx], "grid_id": grid_chunk[grid_idx]}
        )

        base_df = pd.concat([base_df, df_chunk])
        del df_chunk, grid_chunk, date_idx, grid_idx

    print("Sorting data")
    base_df.sort_values(by=["grid_id", "date"], inplace=True)

    return base_df


def main():
    base_df = generate_base_df()

    # Generate and save datasets
    for dataset_name, features in DATASETS.items():
        print(f"Generating dataset: {dataset_name}")
        renamed_features = [f"{dataset_name}_{feature}" for feature in features]
        table = generate_dataset(base_df.copy(), renamed_features)
        save_dataset(table, f"india-map-data-test/data/datasets/{dataset_name}")
        print(f"Dataset {dataset_name} generated and saved.")

        del table
        # Free up memory
        import gc

        print("Freeing up memory")
        gc.collect()


if __name__ == "__main__":
    main()
