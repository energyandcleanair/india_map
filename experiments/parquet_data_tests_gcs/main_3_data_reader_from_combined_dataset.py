import duckdb
import numpy as np
import pandas as pd
from datetime import datetime
import gcsfs
import requests

from google.cloud import storage

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

def get_credentials_for_gcs():
    client = storage.Client()

    # Get the VM's attached service account from the metadata server
    sa_email = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
        headers={"Metadata-Flavor": "Google"}
    ).text

    hmac_key = client.create_hmac_key(sa_email)

    return hmac_key

def read_from_presample(sampled_df):
    con = duckdb.connect()

    print("Read from presample")

    con.query("SET enable_progress_bar = true")

    con.register("sampled", sampled_df)

    credentials = get_credentials_for_gcs()

    con.execute(
        f"""
        CREATE SECRET (
            TYPE gcs,
            KEY_ID '{credentials.access_key}',
            SECRET '{credentials.secret_key}',
        );
        """
    )

    # Single query using USING (date, grid_id)
    query = """
        SELECT *
        FROM sampled
        LEFT JOIN read_parquet('gs://india-map-data-test/data/fully_combined_dataset') AS combined_dataset USING (date, grid_id)
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