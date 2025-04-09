import duckdb
import numpy as np
import pandas as pd
from datetime import datetime
import gcsfs

# This script shows the effectiveness of using DuckDB to read a large single Parquet file
# and sampling data from it (without loading the entire file into memory).

NUMBER_OF_GRIDS = 30_000
YEAR_START = 2015
YEAR_END = 2025

ROWS_TO_SAMPLE = 1_000_000

def generate_sampled_df():
    print("Generating sampled rows directly")

    # Randomly sample dates and grid IDs
    sampled_dates = pd.to_datetime(
        np.random.choice(
            pd.date_range(start=f"{YEAR_START}-01-01", end=f"{YEAR_END}-12-31").values,
            size=ROWS_TO_SAMPLE,
            replace=True
        )
    )
    sampled_grid_ids = np.random.choice(
        [f"grid_{i}" for i in range(NUMBER_OF_GRIDS)],
        size=ROWS_TO_SAMPLE,
        replace=True
    )

    # Create the sampled DataFrame
    sampled_df = pd.DataFrame({
        "date": sampled_dates,
        "grid_id": sampled_grid_ids
    })
    sampled_df['grid_id'] = sampled_df['grid_id'].astype('category')

    return sampled_df

def read_from_presample(sampled_df):
    con = duckdb.connect()

    print("Read from presample")

    con.query("SET enable_progress_bar = true")

    con.register("sampled", sampled_df)

    # Single query using USING (date, grid_id)
    query = """
        SELECT *
        FROM sampled
        LEFT JOIN read_parquet('gs://india-map-data-test/data/combined_dataset-using_joins.parquet') AS combined_dataset USING (date, grid_id)
    """

    start_time = datetime.now()
    print("Running single query join using sampled table...")
    final_df = con.execute(query).df()
    print("Final DataFrame shape:", final_df.shape)
    print(f"Time taken for single query join using sampled table...: {datetime.now() - start_time}")

    con.close()

def main():
    print("Sampling rows")
    # Generate the sampled DataFrame directly
    sampled_df = generate_sampled_df()

    read_from_presample(sampled_df)

if __name__ == '__main__':
    main()