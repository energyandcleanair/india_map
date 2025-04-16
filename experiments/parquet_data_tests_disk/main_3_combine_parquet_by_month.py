import os
import pyarrow.parquet as pq
from tqdm import tqdm

# This demonstrates the effectiveness of using pyarrow to concatenate a large number
# of Parquet files into a single file without loading them all into memory at once.

def combine_parquet_by_month(input_dir: str, output_file: str):
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    # Safely delete the output file if it exists
    try:
        os.remove(output_file)
    except FileNotFoundError:
        pass

    writer = None

    # Iterate through all month directories
    for month_dir in tqdm(sorted(os.listdir(input_dir)), desc="Processing months", unit="month"):
        month_path = os.path.join(input_dir, month_dir)
        if not os.path.isdir(month_path):
            continue

        # Look for the Parquet file in the month directory
        parquet_file = os.path.join(month_path, "data.parquet")
        if not os.path.exists(parquet_file):
            print(f"Parquet file not found in {month_path}. Skipping...")
            continue

        # Read the entire Parquet file
        table = pq.read_table(parquet_file)

        # Write the entire table to the output file incrementally
        if writer is None:
            pq.write_table(table, output_file)
            writer = True
        else:
            pq.write_table(table, output_file, append=True)

    if writer:
        print(f"Data successfully combined into {output_file}")
    else:
        print("No valid Parquet files found to combine.")

def main():
    input_dir = "data/monthly_splits"
    output_file = "data/recombined_dataset.parquet"

    combine_parquet_by_month(input_dir, output_file)

if __name__ == "__main__":
    print("Starting the script...")
    main()
    print("Script execution completed.")
