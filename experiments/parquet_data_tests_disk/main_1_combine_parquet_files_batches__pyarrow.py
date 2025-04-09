import os
import glob
from tqdm import tqdm
from functools import reduce
from pyarrow import parquet as pq
from pyarrow import compute as pc
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor

# This demonstrates the effectiveness of using pyarrow to join a large number of
# Parquet files into a single file without loading them all into memory at once.


def read_partition(dataset_path, grid_id):
    partition_path = os.path.join(dataset_path, f"grid_id={grid_id}")
    if not os.path.exists(partition_path):
        return None
    files = glob.glob(os.path.join(partition_path, "*.parquet"))
    if not files:
        return None

    tables = [
        pq.read_table(f).drop(["__index_level_0__"])
        for f in files
    ]
    table = pa.concat_tables(tables)
    return table


def combine_and_join_by_grid_id(input_dir: str, output_file: str):
    # Delete existing output
    try:
        os.remove(output_file)
    except FileNotFoundError:
        pass

    dataset_names = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]
    all_grid_ids = sorted(
        set(
            os.path.basename(g).split("=")[1]
            for dataset in dataset_names
            for g in glob.glob(os.path.join(input_dir, dataset, "grid_id=*"))
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
                table = read_partition(os.path.join(input_dir, dataset), grid_id)
                if table:
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
        if writer is None:
            writer = pq.ParquetWriter(output_file, combined.schema)
        writer.write_table(combined)

    if writer:
        writer.close()


def main():
    input_dir = "data/datasets"
    output_file = "data/combined_dataset.parquet"
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return
    combine_and_join_by_grid_id(input_dir, output_file)


if __name__ == "__main__":
    print("Starting the script...")
    main()
    print("Script execution completed.")
