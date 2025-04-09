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


def combine_and_join_by_grid_id(input_dir: str, output_file: str):
    fs = gcsfs.GCSFileSystem()

    # Delete existing output
    try:
        fs.rm(output_file)
    except FileNotFoundError:
        pass

    dataset_names = [
        d for d in fs.ls(input_dir) if fs.isdir(f"{input_dir}/{d}")
    ]
    all_grid_ids = sorted(
        set(
            os.path.basename(g).split("=")[1]
            for dataset in dataset_names
            for g in fs.ls(f"{input_dir}/{dataset}")
            if g.startswith("grid_id=")
        )
    )

    batch_size = 10
    writer = None

    for i in tqdm(
        range(0, len(all_grid_ids), batch_size), desc="Processing batches", unit="batch"
    ):
        grid_id_batch = all_grid_ids[i : i + batch_size]

        def read_across_all_datasets_for_grid_id(grid_id):
            grid_tables = []
            for dataset in dataset_names:
                partition_path = f"{input_dir}/{dataset}/grid_id={grid_id}"
                if not fs.exists(partition_path):
                    continue
                files = [f for f in fs.ls(partition_path) if f.endswith(".parquet")]
                if not files:
                    continue

                tables = [
                    pq.read_table(fs.open(f, 'rb')).drop(["__index_level_0__"])
                    for f in files
                ]
                table = pa.concat_tables(tables)
                grid_tables.append(table)

            return reduce(
                lambda a, b: a.join(b, ["grid_id", "date"], join_type="inner"),
                grid_tables,
            )

        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(read_across_all_datasets_for_grid_id, grid_id_batch),
                    desc="Processing grid_ids",
                    unit="grid_id",
                    leave=False,
                    total=len(grid_id_batch),
                )
            )

        combined = pa.concat_tables(results)
        with fs.open(output_file, 'wb' if writer is None else 'ab') as f:
            if writer is None:
                writer = pq.ParquetWriter(f, combined.schema)
            writer.write_table(combined)

    if writer:
        writer.close()


def main():
    input_dir = "gs://india-map-data-test/data/datasets"
    output_file = "gs://india-map-data-test/data/combined_dataset.parquet"
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return
    combine_and_join_by_grid_id(input_dir, output_file)


if __name__ == "__main__":
    print("Starting the script...")
    main()
    print("Script execution completed.")
