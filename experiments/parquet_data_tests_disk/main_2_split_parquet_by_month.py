import os
from pyarrow import parquet as pq
from pyarrow import compute as pc
import pyarrow as pa
from tqdm import tqdm

# This script splits a Parquet file into multiple Parquet files based on the month extracted from a date column.

def split_parquet_by_month(input_file: str, output_dir: str):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writers = {}  # Cache ParquetWriter objects for each month
    reader = pq.ParquetFile(input_file)

    for batch_idx in tqdm(range(reader.num_row_groups), desc="Processing batches", unit="batch"):
        # Read a batch (row group)
        batch = reader.read_row_group(batch_idx)

        if "date" not in batch.column_names:
            print("The dataset does not contain a 'date' column.")
            return

        # Extract the month from the 'date' column
        batch = batch.append_column("month", pc.strftime(batch["date"], format="%Y-%m"))

        # Group by month and write each group to a separate Parquet file
        unique_months = pc.unique(batch["month"]).to_pylist()
        for month in unique_months:
            month_table = batch.filter(pc.equal(batch["month"], month)).drop(["month"])
            month_dir = os.path.join(output_dir, f"month={month}")
            if not os.path.exists(month_dir):
                os.makedirs(month_dir)
            output_file = os.path.join(month_dir, f"data.parquet")

            # Use a ParquetWriter to append data without reading existing files
            if month not in writers:
                writers[month] = pq.ParquetWriter(output_file, month_table.schema)
            writers[month].write_table(month_table)

    # Close all writers
    for writer in writers.values():
        writer.close()

    print(f"Data successfully split into monthly files in {output_dir}")


def main():
    input_file = "data/combined_dataset.parquet"
    output_dir = "data/monthly_splits"

    split_parquet_by_month(input_file, output_dir)


if __name__ == "__main__":
    print("Starting the script...")
    main()
    print("Script execution completed.")
