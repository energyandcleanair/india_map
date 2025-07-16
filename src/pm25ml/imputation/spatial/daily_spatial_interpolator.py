"""Provide functionality for spatially imputing data using grid-based interpolation."""

import re

import numpy as np
import polars as pl
from scipy.interpolate import griddata

from pm25ml.collectors.grid_loader import Grid
from pm25ml.logging import logger


class DailySpatialInterpolator:
    """
    Impute missing values in a DataFrame using spatial interpolation.

    It will use linear interpolation for points with valid neighbors,
    and nearest neighbour interpolation for points without valid neighbors.
    """

    def __init__(self, *, grid: Grid, value_column_regex_selector: str) -> None:
        """
        Initialize the imputer with a grid and a regex for selecting columns.

        :param grid: The grid containing original coordinates.
        :param value_column_regex_selector: A regex pattern to select columns for imputation.
        """
        self.grid = grid
        self.value_column_regex_selector = value_column_regex_selector

    def impute(
        self,
        # We will update this DataFrame "in place".
        df_out: pl.DataFrame,
    ) -> pl.DataFrame:
        """Perform spatial imputation for a given dataframe."""
        required_columns = ["grid_id", "date"]
        for col in required_columns:
            if col not in df_out.columns:
                msg = f"DataFrame is missing required column: {col}"
                raise ValueError(msg)

        unique_dates = df_out.select("date").unique().to_series().sort()

        df_out = df_out.join(
            self.grid.df_original,
            on="grid_id",
            how="left",
        )
        regex = re.compile(self.value_column_regex_selector)
        value_cols = [col for col in df_out.columns if regex.match(col)]

        for date in unique_dates:
            logger.debug(f"Imputing data for date: {date}")
            daily_df = df_out.filter(pl.col("date") == date)

            for col in value_cols:
                # We convert the column to a NumPy array so we can modify it in place.
                all_values = daily_df[col].to_numpy(writable=True)
                all_points = np.stack([daily_df["original_x"], daily_df["original_y"]], axis=1)

                # This is a mask we're going to use to select for valid points, and inverse it to
                # select for NaN points.
                valid_mask = ~np.isnan(all_values)

                # When using griddata, we use numpy index masking to assign to only the missing
                # points by matching the index mask in assignment and the selection for the `xi`
                # argument (the points to interpolate for).
                #
                # griddata maintains the order of points, so we can directly assign the results back
                # to the original array.
                #
                # We only use original valid points for interpolation in both stages.
                #
                # We need to handle two stages of interpolation because some values are outside
                # the convex hull of the valid points and the linear interpolation will fail for
                # this.

                # Replace NaN values using linear interpolation first.
                all_values[~valid_mask] = griddata(
                    points=all_points[valid_mask],
                    values=all_values[valid_mask],
                    xi=all_points[~valid_mask],
                    method="linear",
                )
                # For any remaining NaN values, use nearest neighbour interpolation.
                nan_mask = np.isnan(all_values)
                all_values[nan_mask] = griddata(
                    points=all_points[~nan_mask],
                    values=all_values[~nan_mask],
                    xi=all_points[nan_mask],
                    method="nearest",
                )

                # We want to overwrite just this column for the current date, the easiest way
                # to do this, is to extract the column as a NumPy array, modify it by a mask,
                # and then assign it back to the DataFrame. Mixing polars and NumPy here is hard.
                date_mask = df_out["date"] == date
                new_vals = df_out[col].to_numpy(writable=True)
                new_vals[date_mask] = all_values

                # Now we assign the modified values back to the DataFrame.
                df_out = df_out.with_columns(
                    pl.Series(
                        name=col,
                        values=new_vals,
                    ),
                )

        # We want to drop the original coordinates, as they are no longer needed.
        columns_to_drop = [col for col in self.grid.df_original.columns if col != "grid_id"]
        return df_out.sort(["date", "grid_id"]).drop(
            columns_to_drop,
        )
