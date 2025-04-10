import os
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm
import gcsfs

# This demonstrates the effectiveness of using pyarrow to concatenate a large number
# of Parquet files into a single file without loading them all into memory at once.

def combine_parquet_by_month(input_dir: str, output_file: str):
    fs = gcsfs.GCSFileSystem()

    # Check if input directory exists
    if not fs.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    # Safely delete the output file if it exists
    try:
        if fs.exists(output_file):
            fs.rm(output_file)
    except FileNotFoundError:
        pass

    # Iterate through all month directories
    month_dirs = [d for d in fs.ls(input_dir, detail=True) if d['type'] == 'directory']
    if not month_dirs:
        print("No month directories found.")
        raise FileNotFoundError("No month directories found.")

    all_datasets = []

    for month_dir in tqdm(sorted(month_dirs, key=lambda x: x['name']), desc="Processing months", unit="month"):
        month_path = month_dir['name']

        # Look for the Parquet file in the month directory
        parquet_file = f"{month_path}/data.parquet"
        if not fs.exists(parquet_file):
            print(f"Parquet file not found in {month_path}. Skipping...")
            continue

        # Load the dataset instead of individual tables
        dataset = ds.dataset(parquet_file, filesystem=fs, format="parquet")
        all_datasets.append(dataset)

    if not all_datasets:
        print("No valid Parquet datasets found to combine.")
        return

    # Concatenate all datasets and write them to the output file
    combined_dataset = ds.dataset(all_datasets)

    parquet_format = ds.ParquetFileFormat()

    file_options = parquet_format.make_write_options(compression="snappy")

    ds.write_dataset(
        combined_dataset,
        output_file,
        filesystem=fs,
        format=parquet_format,
        file_options=file_options,
    )

    print(f"Data successfully combined into {output_file}")

def main():
    input_dir = "india-map-data-test/data/combined_monthly"
    output_file = "india-map-data-test/data/fully_combined_dataset.parquet"

    combine_parquet_by_month(input_dir, output_file)

if __name__ == "__main__":
    print("Starting the script...")
    main()
    print("Script execution completed.")
