import duckdb
import numpy as np
import pandas as pd
from datetime import datetime
import gcsfs
import requests
from fsspec import filesystem

from google.cloud import storage
import psycopg2
from sqlalchemy import create_engine, text

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
    db_url = "postgresql://admin:admin_password@localhost:5432/my_database"
    engine = create_engine(db_url)

    print("Read from presample")

    with engine.connect() as connection:
        # Create a temporary table for the sampled data
        print("Creating temporary table for sampled data...")
        connection.execute(text("""
            DROP TABLE IF EXISTS sampled;
        """))
        connection.execute(text("""
            CREATE TABLE sampled (
                date DATE NOT NULL,
                grid_id VARCHAR(50) NOT NULL
            )
        """))
        connection.execute(text(f"""
            CREATE INDEX idx_sampled_date ON sampled (date);
            CREATE INDEX idx_sampled_grid_id ON sampled (grid_id);
            CREATE INDEX idx_sampled_date_grid_id ON sampled (date, grid_id);
        """))
        connection.commit()

        # Insert sampled data into the temporary table
        print("Inserting sampled data into temporary table...")
        sampled_df.to_sql('sampled', con=engine, if_exists='append', index=False)

        # Perform the join query
        query = """
            SELECT *
            FROM sampled
            JOIN crea_pm25 USING (grid_id, date)
            JOIN gee_era5 USING (grid_id, date)
            JOIN gee_modis USING (grid_id, date)
            JOIN gee_usgs USING (grid_id, date)
            JOIN generated_date USING (grid_id, date)
            JOIN generated_nasa_earthdata USING (grid_id, date)
            JOIN generated_weather USING (grid_id, date)
            JOIN imputed_aod USING (grid_id, date)
            JOIN imputed_co2 USING (grid_id, date)
            JOIN imputed_no2 USING (grid_id, date)
            JOIN nasa_earthdata USING (grid_id, date)
        """

        print("Running single query join using sampled table...")
        start_time = datetime.now()
        result = connection.execute(text(query))
        final_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        print("Final DataFrame shape:", final_df.shape)
        print(f"Time taken for single query join using sampled table...: {datetime.now() - start_time}")

def main():
    print("Sampling rows")
    # Generate the sampled DataFrame directly
    sampled_df = generate_sampled_df()

    read_from_presample(sampled_df)

if __name__ == '__main__':
    main()