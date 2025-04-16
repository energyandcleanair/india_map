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
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text
import os

# This script generates synthetic datasets for a specified range of years and grid IDs.
# Used for testing and benchmarking purposes.

NUMBER_OF_GRIDS = 30000
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


def execute_sql_in_database(dataset_name: str, features: list[str]):
    """
    Executes the generated SQL in the PostgreSQL database.
    Drops the table if it already exists, inserts data, and adds indices.
    """
    db_url = "postgresql://admin:admin_password@localhost:5432/my_database"
    engine = create_engine(db_url)

    drop_table_sql = f"DROP TABLE IF EXISTS {dataset_name};"

    create_table_sql = f"""
    CREATE TABLE {dataset_name} (
        date DATE NOT NULL,
        grid_id VARCHAR(50) NOT NULL,
        {', '.join([f'{feature} DOUBLE PRECISION' for feature in features])},
        PRIMARY KEY (date, grid_id)
    );
    """

    feature_columns = ', '.join(features)
    feature_values = ',\n    '.join(["(grid_id * extract(doy from d)::int % 10000)::float / 10000" for _ in features])

    insert_sql = f"""
    INSERT INTO {dataset_name} (grid_id, date, {feature_columns})
    SELECT
        'grid_' || grid_id AS grid_id,
        d::date,
        {feature_values}
    FROM generate_series(0, {NUMBER_OF_GRIDS - 1}) AS grid_id
    CROSS JOIN generate_series('{YEAR_START}-01-01'::date, '{YEAR_END}-12-31'::date, '1 day') AS d;
    """

    add_indices_sql = f"""
        CREATE INDEX idx_{dataset_name}_date ON {dataset_name} (date);
        CREATE INDEX idx_{dataset_name}_grid_id ON {dataset_name} (grid_id);
        CREATE INDEX idx_{dataset_name}_date_grid_id ON {dataset_name} (date, grid_id);
    """

    with engine.connect() as connection:
        with connection.begin():
            tqdm.write(f"Dropping old table: {dataset_name}")
            connection.execute(text(drop_table_sql))
            tqdm.write(f"Creating new table: {dataset_name}")
            connection.execute(text(create_table_sql))
            tqdm.write(f"Inserting data into table: {dataset_name}")
            connection.execute(text(insert_sql))
            tqdm.write(f"Adding indices to table: {dataset_name}")
            connection.execute(text(add_indices_sql))
            tqdm.write(f"Committing changes to database.")

def main():
    for dataset_name, features in tqdm(DATASETS.items(), unit="dataset"):
        tqdm.write(f"Generating SQL for dataset: {dataset_name}")
        renamed_features = [f"{dataset_name}_{feature}" for feature in features]
        execute_sql_in_database(dataset_name, renamed_features)
        tqdm.write(f"SQL for dataset {dataset_name} executed in the database.")

if __name__ == "__main__":
    main()
