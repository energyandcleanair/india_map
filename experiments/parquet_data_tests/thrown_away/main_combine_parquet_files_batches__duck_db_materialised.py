import glob
import duckdb
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds
import os
from collections import defaultdict
from functools import reduce
from tqdm import tqdm

def combine_and_join_by_grid_id(output_file: str):
    # List of directories containing hive formatted parquet databases
    directories = [
        "data/datasets/crea_pm25",
        "data/datasets/gee_era5",
        "data/datasets/gee_modis",
        "data/datasets/gee_usgs",
        "data/datasets/generated_date",
        "data/datasets/generated_nasa_earthdata",
        "data/datasets/generated_weather",
        "data/datasets/imputed_aod",
        "data/datasets/imputed_co2",
        "data/datasets/imputed_no2",
        "data/datasets/nasa_earthdata",
    ]

    # Initialize a DuckDB connection
    con = duckdb.connect()

    # Iterate over each directory and import the hive formatted parquet databases
    for directory in tqdm(directories, desc="Importing Parquet Databases"):
        
        # Read the hive formatted parquet dataset into DuckDB
        con.execute(f"CREATE TABLE {os.path.basename(directory)} AS SELECT * FROM parquet_scan('{directory}/**/*.parquet', hive_partitioning=TRUE)")
        print(f"Successfully imported {directory} into DuckDB.")

    # Add an index to each table for grid_id and date
    for directory in directories:
        table_name = os.path.basename(directory)
        index_name = f"{table_name}_grid_id_date_idx"
        con.execute(f"CREATE INDEX {index_name} ON {table_name} (grid_id, date)")
        print(f"Index '{index_name}' created on {table_name} for grid_id and date.")
    
    # Perform the join operation across all tables
    table_names = [os.path.basename(directory) for directory in directories]
    joined_table = None
    for table_name in tqdm(table_names, desc="Joining Tables"):
        if joined_table is None:
            joined_table = con.table(table_name)
        else:
            joined_table = joined_table.join(con.table(table_name), ["grid_id", "date"], "inner")
    
    # Export the combined table to a Parquet file
    try:
        joined_table.write_parquet(output_file)
        print(f"Successfully exported the combined table to {output_file}.")
    except Exception as e:
        print(f"Failed to export the combined table: {e}")

    print(f"Combined DuckDB database exported to {output_file}.")

def main():
    output_file = "data/combined_dataset.parquet"  # Output file path
    combine_and_join_by_grid_id(output_file)

if __name__ == "__main__":
    print("Starting the script...")
    main()
    print("Script execution completed.")
