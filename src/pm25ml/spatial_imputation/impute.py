"""Provide functionality for spatially imputing data using grid-based interpolation."""

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from dependency_injector.wiring import Provide, inject
from scipy.interpolate import griddata

from pm25ml.collectors.grid_loader import Grid
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.logging import logger
from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env


@inject
def _main(
    *,
    grid: Grid = Provide[Pm25mlContainer.in_memory_grid],
    combined_storage: CombinedStorage = Provide[Pm25mlContainer.combined_storage],
    column_regex: str,
) -> None:
    """Perform spatial imputation for each month."""
    original_grid = grid.df_original

    ds = combined_storage.scan_stage("combined_monthly").select(
        "month",
        "grid_id",
        "date",
        pl.col(column_regex),
    )

    months = ds.select("month").unique().collect(engine="streaming").to_series().sort()

    months_to_upload = [
        month
        for month in months
        if not combined_storage.does_dataset_exist(f"stage=era5_spatially_imputed/month={month}")
    ]

    logger.info(
        f"Found {len(months_to_upload)} months to process for spatial imputation.",
    )

    def process_month(month: str) -> None:
        """Process spatial imputation for a specific month."""
        logger.debug(f"Spatially imputing data for month: {month}")

        ds_name = f"stage=era5_spatially_imputed/month={month}"
        if combined_storage.does_dataset_exist(ds_name):
            logger.debug(f"Dataset for month {month} already exists, skipping.")
            return

        month_df = ds.filter(pl.col("month") == month).collect(engine="streaming")

        expected_length = month_df.select(pl.len()).to_series()[0]

        imputed_month = spatially_impute_month(month_df, original_grid, column_regex).select(
            "grid_id",
            "date",
            pl.col(column_regex),
        )

        actual_length = imputed_month.select(pl.len()).to_series()[0]

        if actual_length != expected_length:
            msg = f"Imputed month {month} has length {actual_length}, expected {expected_length}."
            raise ValueError(msg)

        combined_storage.write_to_destination(
            imputed_month,
            f"stage=era5_spatially_imputed/month={month}",
        )

    with ThreadPoolExecutor(8) as executor:
        executor.map(process_month, months_to_upload)


def spatially_impute_month(
    month_df: pl.DataFrame,
    original_grid: pl.DataFrame,
    column_regex: str,
) -> pl.DataFrame:
    """Perform spatial imputation for a given month."""
    unique_dates = month_df.select("date").unique().to_series().sort()

    df_out = month_df.join(
        original_grid,
        on="grid_id",
        how="left",
    ).rename(
        {
            "original_x": "x",
            "original_y": "y",
        },
    )
    regex = re.compile(column_regex)
    value_cols = [col for col in df_out.columns if regex.match(col)]

    for date in unique_dates:
        logger.debug(f"Imputing data for date: {date}")
        daily_df = df_out.filter(pl.col("date") == date)

        for col in value_cols:
            all_values = daily_df[col].to_numpy(writable=True)
            all_points = np.stack([daily_df["x"], daily_df["y"]], axis=1)

            valid_mask = ~np.isnan(all_values)

            all_values[~valid_mask] = griddata(
                all_points[valid_mask],
                all_values[valid_mask],
                all_points[~valid_mask],
                method="linear",
            )
            nan_mask = np.isnan(all_values)
            all_values[nan_mask] = griddata(
                all_points[~nan_mask],
                all_values[~nan_mask],
                all_points[nan_mask],
                method="nearest",
            )

            new_vals = df_out[col].to_numpy(writable=True)
            new_vals[df_out["date"] == date] = all_values

            # This assigns the imputed values back to the DataFrame
            df_out = df_out.with_columns(
                pl.Series(
                    name=col,
                    values=new_vals,
                ),
            )

    return df_out.sort(["date", "grid_id"]).select(
        "month",
        "grid_id",
        "date",
        pl.col(column_regex),
    )


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main(
        column_regex=r"^era5_land__.*$",
    )
