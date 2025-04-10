import os
import glob
from tqdm import tqdm
from functools import reduce
from pyarrow import parquet as pq
from pyarrow import compute as pc
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor
import gcsfs

# This demonstrates the effectiveness of using pyarrow to join a large number of
# Parquet files into a single file without loading them all into memory at once.


def combine_and_join_by_month(input_dir: str, output_dir: str):
    fs = gcsfs.GCSFileSystem()

    # Delete existing output directory if it exists
    try:
        if fs.exists(output_dir):
            fs.rm(output_dir, recursive=True)
    except FileNotFoundError:
        pass

    dataset_names = [
        os.path.basename(d["name"]) for d in fs.ls(input_dir, detail=True) if d["type"] == "directory"
    ]

    print(f"Found datasets: {dataset_names}")
    if not dataset_names:
        print("No datasets found.")
        raise FileNotFoundError("No datasets found.")
    
    print(f"Found {len(dataset_names)} datasets.")

    dataset_dirs = [
        f"{input_dir}/{dataset}" for dataset in dataset_names
    ]

    all_months = sorted(
        set(
            os.path.basename(m).split("=")[1]
            for dataset_dir in dataset_dirs
            for m in fs.ls(dataset_dir)
            if os.path.basename(m).startswith("month=")
        )
    )

    if not all_months:
        print("No months found.")
        raise FileNotFoundError("No months found.")
    
    print(f"Found months: {all_months}")

    for month in tqdm(all_months, desc="Processing months", unit="month"):
        month_tables = []
        tqdm.write(f"Processing month: {month}")
        for dataset in dataset_names:
            tqdm.write(f"Reading dataset: {dataset}")
            partition_path = f"{input_dir}/{dataset}/month={month}"
            if not fs.exists(partition_path):
                continue
            files = [f for f in fs.ls(partition_path) if f.endswith(".parquet")]
            if not files:
                continue

            tables = []
            for f in files:
                table = pq.read_table(fs.open(f, 'rb'))
                if "__index_level_0__" in table.column_names:
                    table = table.drop(["__index_level_0__"])
                tables.append(table)

            table = pa.concat_tables(tables)
            month_tables.append(table)

        if month_tables:
            tqdm.write(f"Joining tables for month: {month}")
            combined = reduce(
                lambda a, b: a.join(b, ["grid_id", "date"], join_type="inner"),
                month_tables,
            )

            tqdm.write(f"Writing combined table for month: {month}")
            month_output_path = f"{output_dir}/month={month}/part-0.parquet"
            fs.makedirs(os.path.dirname(month_output_path), exist_ok=True)
            with fs.open(month_output_path, 'wb') as f:
                writer = pq.ParquetWriter(f, combined.schema)
                writer.write_table(combined)
                writer.close()

def main():
    input_dir = "india-map-data-test/data/datasets"
    output_dir = "india-map-data-test/data/combined_monthly"
    combine_and_join_by_month(input_dir, output_dir)


if __name__ == "__main__":
    print("Starting the script...")
    main()
    print("Script execution completed.")
