import os
import pyarrow.parquet as pq
from tqdm import tqdm
import gcsfs

# This demonstrates the effectiveness of using pyarrow to concatenate a large number
# of Parquet files into a single file without loading them all into memory at once.

def combine_parquet_by_month(input_dir: str, output_file: str):
    fs = gcsfs.GCSFileSystem()

    if not fs.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    # Safely delete the output file if it exists
    try:
        fs.rm(output_file)
    except FileNotFoundError:
        pass

    writer = None

    # Iterate through all month directories
    for month_dir in tqdm(sorted(fs.ls(input_dir)), desc="Processing months", unit="month"):
        month_path = f"{input_dir}/{month_dir}"
        if not fs.isdir(month_path):
            continue

        # Look for the Parquet file in the month directory
        parquet_file = f"{month_path}/data.parquet"
        if not fs.exists(parquet_file):
            print(f"Parquet file not found in {month_path}. Skipping...")
            continue

        # Read the entire Parquet file
        with fs.open(parquet_file, 'rb') as f:
            table = pq.read_table(f)

        # Write the entire table to the output file incrementally
        with fs.open(output_file, 'wb' if writer is None else 'ab') as f:
            if writer is None:
                pq.write_table(table, f)
                writer = True
            else:
                pq.write_table(table, f, append=True)

    if writer:
        print(f"Data successfully combined into {output_file}")
    else:
        print("No valid Parquet files found to combine.")

def main():
    input_dir = "gs://india-map-data-test/data/monthly_splits"
    output_file = "gs://india-map-data-test/data/recombined_dataset.parquet"

    combine_parquet_by_month(input_dir, output_file)

if __name__ == "__main__":
    print("Starting the script...")
    main()
    print("Script execution completed.")
