import functools
import os
from pathlib import Path
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
    dataset_names: list[str] = [
        Path(d["name"]).name for d in fs.ls(input_dir, detail=True) if d["type"] == "directory"
    ]

    print(f"Found datasets: {dataset_names}")
    if not dataset_names:
        print("No datasets found.")
        raise FileNotFoundError("No datasets found.")
    print(f"Found {len(dataset_names)} datasets.")

    dataset_dirs = [f"{input_dir}/{dataset}" for dataset in dataset_names]
    all_months = sorted(
        {
            Path(m).name.split("=")[1]
            for dataset_dir in dataset_dirs
            for m in fs.ls(dataset_dir)
            if Path(m).name.startswith("month=")
        },
    )

    if not all_months:
        msg = "No months found."
        print(msg)  # noqa: T201
        raise FileNotFoundError(msg)

    print(f"Found months: {all_months}")

    for month in tqdm(all_months, desc="Processing months", unit="month"):
        tqdm.write(f"Processing month: {month}")

        def process_dataset(dataset: str, month: str) -> pa.Table:
            """Combine all Parquet files for a given dataset and month into a single table."""
            tqdm.write(f"Reading dataset: {dataset}")
            partition_path = f"{input_dir}/{dataset}/month={month}"
            if not fs.exists(partition_path):
                return None
            files = [f for f in fs.ls(partition_path) if f.endswith(".parquet")]
            if not files:
                return None

            tables = []
            for f in files:
                table = pq.read_table(fs.open(f, "rb"))
                if "__index_level_0__" in table.column_names:
                    table = table.drop(["__index_level_0__"])
                tables.append(table)

            return pa.concat_tables(tables)

        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(functools.partial(process_dataset, month=month), dataset_names),
                    total=len(dataset_names),
                    desc="Processing datasets",
                    unit="dataset",
                ),
            )

        month_tables = [result for result in results if result is not None]

        if month_tables:
            tqdm.write(f"Joining tables for month: {month}")

            def join_two_tables(left_table: pa.Table, right_table: pa.Table) -> pa.Table:
                return left_table.join(
                    right_table,
                    keys=["grid_id", "date"],
                    join_type="full outer",
                )

            # Function to manage the parallel joining process
            def parallel_outer_join(tables: list[pa.Table]) -> pa.Table:
                with ThreadPoolExecutor() as executor:
                    # Initial join operations
                    while len(tables) > 1:
                        # Pair tables for joining
                        table_pairs = list(zip(tables[0::2], tables[1::2]))
                        # Perform joins in parallel
                        tables = list(executor.map(lambda p: join_two_tables(*p), table_pairs))
                        # If there's an odd table out, append it to the list
                        if len(tables) % 2 == 1:
                            tables.append(tables.pop(0))
                return tables[0]

            # Performing the parallel full outer join
            combined = parallel_outer_join(month_tables)

            tqdm.write(f"Writing combined table for month: {month}")
            month_output_path = f"{output_dir}/month={month}/part-0.parquet"
            fs.makedirs(os.path.dirname(month_output_path), exist_ok=True)
            with fs.open(month_output_path, "wb") as f:
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
