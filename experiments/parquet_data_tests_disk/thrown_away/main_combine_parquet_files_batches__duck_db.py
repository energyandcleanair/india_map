import glob
import duckdb
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds
import os
from collections import defaultdict
from functools import reduce
from tqdm import tqdm

def combine_and_join_by_grid_id(input_dir: str, output_file: str):

    # Step 1: Delete the output file if it already exists
    try:
        os.remove(output_file)
    except FileNotFoundError:
        pass
    dataset_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    all_grid_ids = sorted(set(
        os.path.basename(g).split("=")[1]
        for dataset in dataset_names
        for g in glob.glob(os.path.join(input_dir, dataset, "grid_id=*"))
    ))

    # Step 2: Use DuckDB to process grid_id partitions in batches of 10
    con = duckdb.connect(database=":memory:")

    batch_size = 100
    for i in tqdm(range(0, len(all_grid_ids), batch_size), desc="Processing batches", unit="batch"):
        grid_id_batch = all_grid_ids[i:i + batch_size]

        # Load all partitions for this batch of grid_ids across datasets
        tables = []
        for grid_id in tqdm(grid_id_batch, desc="Processing grid_ids", unit="grid_id", leave=False):
            grid_tables = []
            for dataset in dataset_names:
                partition_path = os.path.join(input_dir, dataset, f"grid_id={grid_id}")
                if not os.path.exists(partition_path):
                    continue

                # Read Parquet via DuckDB
                rel = (
                    con
                    .from_parquet(os.path.join(partition_path, "*.parquet"))
                    .project(f"*, '{grid_id}' AS grid_id")
                )
                grid_tables.append(rel)
            
            batch_joined = reduce(
                lambda a, b: a.join(b, ["date"], how="inner"),
                grid_tables
            )
            tables.append(batch_joined)

        tqdm.write("Unioning batch")
        joined = pa.concat_tables(tables)

        tqdm.write("Writing batch to Parquet")
        # Append to output Parquet file
        joined.to_parquet(output_file, append=True)

def main():
    input_dir = "data/datasets"  # Directory containing the Hive-partitioned Parquet files
    output_file = "data/combined_dataset.parquet"  # Output file path

    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    combine_and_join_by_grid_id(input_dir, output_file)

if __name__ == "__main__":
    print("Starting the script...")
    main()
    print("Script execution completed.")
