
import duckdb
import numpy as np
import pandas as pd
from datetime import datetime


NUMBER_OF_GRIDS = 30_000
YEAR_START = 2015
YEAR_END = 2025

ROWS_TO_SAMPLE = 1_000_000

def generate_base_df():
  
  print("Generating date range")
  # Create a date range for the years and months
  date_range = pd.date_range(start=f"{YEAR_START}-01-01", end=f"{YEAR_END}-12-31", freq="D")
  
  print("Generating grid IDs")
  # Create a grid of grid IDs
  grid_ids = [f"grid_{i}" for i in range(NUMBER_OF_GRIDS)]
  
  print("Generating date*grid_id combinations")
  # Create a DataFrame with all combinations of date and grid ID
  base_df = pd.DataFrame({
      "date": np.repeat(date_range.values, len(grid_ids)),
      "grid_id": np.tile(grid_ids, len(date_range))
  })
  # Ensure correct types for date and grid_id
  base_df['date'] = pd.to_datetime(base_df['date'])
  base_df['grid_id'] = base_df['grid_id'].astype('category')

  return base_df

def read_from_presample(sampled_df):
  con = duckdb.connect()

  con.query("SET enable_progress_bar = true")

  con.register("sampled", sampled_df)

  # Single query using USING (date, grid_id)
  query = """
      SELECT *
      FROM sampled
      LEFT JOIN read_parquet('data/crea_pm25/*/*.parquet') AS crea_pm25 USING (date, grid_id)
      LEFT JOIN read_parquet('data/gee_era5/*/*.parquet') AS gee_era5 USING (date, grid_id)
      LEFT JOIN read_parquet('data/gee_modis/*/*.parquet') AS gee_modis USING (date, grid_id)
      LEFT JOIN read_parquet('data/gee_usgs/*/*.parquet') AS gee_usgs USING (date, grid_id)
      LEFT JOIN read_parquet('data/generated_date/*/*.parquet') AS generated_date USING (date, grid_id)
      LEFT JOIN read_parquet('data/generated_nasa_earthdata/*/*.parquet') AS generated_nasa_earthdata USING (date, grid_id)
      LEFT JOIN read_parquet('data/generated_weather/*/*.parquet') AS generated_weather USING (date, grid_id)
      LEFT JOIN read_parquet('data/imputed_aod/*/*.parquet') AS imputed_aod USING (date, grid_id)
      LEFT JOIN read_parquet('data/imputed_co2/*/*.parquet') AS imputed_co2 USING (date, grid_id)
      LEFT JOIN read_parquet('data/imputed_no2/*/*.parquet') AS imputed_no2 USING (date, grid_id)
      LEFT JOIN read_parquet('data/nasa_earthdata/*/*.parquet') AS nasa_earthdata USING (date, grid_id)
  """

  start_time = datetime.now()
  print("Running single query join using sampled table...")
  final_df = con.execute(query).df()
  print("Final DataFrame shape:", final_df.shape)
  print(f"Time taken for single query join using sampled table...: {datetime.now() - start_time}")

  con.close()
  
def stream_and_sample_through(sampled_df):
  con = duckdb.connect()
  con.query("SET enable_progress_bar = true")

  query = """
      SELECT *
      FROM read_parquet('data/crea_pm25/*/*.parquet') AS crea_pm25
      LEFT JOIN read_parquet('data/gee_era5/*/*.parquet') AS gee_era5 USING (date, grid_id)
      LEFT JOIN read_parquet('data/gee_modis/*/*.parquet') AS gee_modis USING (date, grid_id)
      LEFT JOIN read_parquet('data/gee_usgs/*/*.parquet') AS gee_usgs USING (date, grid_id)
      LEFT JOIN read_parquet('data/generated_date/*/*.parquet') AS generated_date USING (date, grid_id)
      LEFT JOIN read_parquet('data/generated_nasa_earthdata/*/*.parquet') AS generated_nasa_earthdata USING (date, grid_id)
      LEFT JOIN read_parquet('data/generated_weather/*/*.parquet') AS generated_weather USING (date, grid_id)
      LEFT JOIN read_parquet('data/imputed_aod/*/*.parquet') AS imputed_aod USING (date, grid_id)
      LEFT JOIN read_parquet('data/imputed_co2/*/*.parquet') AS imputed_co2 USING (date, grid_id)
      LEFT JOIN read_parquet('data/imputed_no2/*/*.parquet') AS imputed_no2 USING (date, grid_id)
      LEFT JOIN read_parquet('data/nasa_earthdata/*/*.parquet') AS nasa_earthdata USING (date, grid_id)
      ORDER BY grid_id, date
    """
  
  start_time = datetime.now()
  print("Running streaming query")
  stream = con.execute(query)

  sampled_rows = pd.DataFrame()

  chunk_size = 100_000
  current_chunk = 0
  total_chunks_expected = (NUMBER_OF_GRIDS * (YEAR_END - YEAR_START + 1)) / chunk_size
  for chunk in stream.iter_batches(batch_size=chunk_size):
      print(f"Sampling chunk {current_chunk} of {total_chunks_expected}")
      current_chunk += 1
      # Convert the chunk to a DataFrame
      chunk_df = pd.DataFrame(chunk)
      # Sample the chunk
      sampled_chunk = chunk_df.merge(sampled_df, on=["date", "grid_id"], how="inner")

      # Append the sampled chunk to the final DataFrame
      sampled_rows = pd.concat([sampled_rows, sampled_chunk], ignore_index=True)
  
  print("Final DataFrame shape:", sampled_rows.shape)
  print(f"Time taken for streaming query: {datetime.now() - start_time}")
  con.close()

def main():
  base_df = generate_base_df()

  print("Sampling rows")
  # Sample a subset of the DataFrame
  sampled_df = base_df.sample(n=ROWS_TO_SAMPLE, random_state=42).reset_index(drop=True)

  stream_and_sample_through(sampled_df)
  read_from_presample(sampled_df)
  

if __name__ == '__main__':
  main()