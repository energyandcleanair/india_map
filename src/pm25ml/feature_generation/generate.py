"""Feature generation for PM2.5 data."""

import math

import polars as pl

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.logging import logger
from pm25ml.setup.date_params import TemporalConfig

ABSOLUTE_ZERO = 273.15
MAGNUS_APPROXIMATION_A = 17.625
MAGNUS_APPROXIMATION_B = 234.04
MONSOON_SEASON_MONTHS = [6, 7, 8, 9]  # June to September


class FeatureGenerator:
    """Class to generate features for PM2.5 data."""

    def __init__(
        self,
        combined_storage: CombinedStorage,
        temporal_config: TemporalConfig,
        input_data_artifact: DataArtifactRef,
        output_data_artifact: DataArtifactRef,
    ) -> None:
        """Initialize the FeatureGenerator."""
        self.combined_storage = combined_storage
        self.temporal_config = temporal_config
        self.input_data_artifact = input_data_artifact
        self.output_data_artifact = output_data_artifact

    def generate(self) -> None:
        """Generate features for PM2.5 data."""
        lf = self.combined_storage.scan_stage(self.input_data_artifact.stage)
        for year in self.temporal_config.years:
            logger.info(f"Generating features for year: {year}")

            # This window must include the current year and the previous year
            months_in_window = [
                month.format("YYYY-MM")
                for month in self.temporal_config.months
                if month.year in (year, year - 1)
            ]

            dewpoint_c = pl.col("era5_land__dewpoint_temperature_2m") - ABSOLUTE_ZERO
            temperature_c = pl.col("era5_land__temperature_2m") - ABSOLUTE_ZERO

            relative_humidity: pl.Expr = (
                MAGNUS_APPROXIMATION_A * dewpoint_c / (MAGNUS_APPROXIMATION_B + dewpoint_c)
                - MAGNUS_APPROXIMATION_A * temperature_c / (MAGNUS_APPROXIMATION_B + temperature_c)
            ).exp()

            wind_degree: pl.Expr = (
                pl.arctan2(
                    -pl.col("era5_land__u_component_of_wind_10m"),
                    -pl.col("era5_land__v_component_of_wind_10m"),
                )
                * 180.0
                / math.pi
                + 360.0
            ) % 360.0

            monsoon_season: pl.Expr = (
                pl.when(pl.col("date").dt.month().is_in(MONSOON_SEASON_MONTHS))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
            )

            def weekly_rolling_mean(col_name: str) -> pl.Expr:
                return (
                    pl.col(col_name)
                    .fill_nan(None)
                    .rolling_mean(7, min_samples=1)
                    .backward_fill()
                    .forward_fill()
                    .over("grid_id")
                )

            def annual_rolling_mean(col_name: str) -> pl.Expr:
                return (
                    pl.col(col_name)
                    .fill_nan(None)
                    .rolling_mean(365, min_samples=1)
                    .backward_fill()
                    .forward_fill()
                    .over("grid_id")
                )

            def annual_average(col_name: str) -> pl.Expr:
                return pl.col(col_name).fill_nan(None).mean().over(["grid_id", "year"])

            def all_averages(col_name: str) -> dict[str, pl.Expr]:
                return {
                    f"{col_name}__mean_r7d": weekly_rolling_mean(col_name),
                    f"{col_name}__mean_r365d": annual_rolling_mean(col_name),
                    f"{col_name}__mean_year": annual_average(col_name),
                    f"{col_name}__mean_all": pl.col(col_name).fill_nan(None).mean().over("grid_id"),
                }

            lf_for_year = (
                lf.filter(pl.col("month").is_in(months_in_window))
                .with_columns(
                    pl.col("date").cast(pl.Date),
                )
                .sort(
                    [
                        "month",
                        "date",
                        "grid_id",
                    ],
                )
                .with_columns(
                    pl.col("date").dt.year().alias("year"),
                )
                .with_columns(
                    day_of_year=pl.col("date").dt.ordinal_day(),
                    era5_land__relative_humidity_computed=relative_humidity,
                    era5_land__wind_degree_computed=wind_degree,
                )
                .with_columns(
                    **all_averages("merra_aot__aot"),
                    **all_averages("merra_co__co"),
                    **all_averages("merra_co_top__co"),
                    **all_averages("era5_land__temperature_2m"),
                    **all_averages("era5_land__dewpoint_temperature_2m"),
                    **all_averages("era5_land__relative_humidity_computed"),
                    **all_averages("era5_land__wind_degree_computed"),
                    **all_averages("era5_land__u_component_of_wind_10m"),
                    **all_averages("era5_land__v_component_of_wind_10m"),
                    **all_averages("era5_land__total_precipitation_sum"),
                    **all_averages("era5_land__surface_net_thermal_radiation_sum"),
                    **all_averages("era5_land__surface_pressure"),
                    **all_averages("era5_land__leaf_area_index_high_vegetation"),
                    **all_averages("era5_land__leaf_area_index_low_vegetation"),
                    **all_averages("omi_no2__no2"),
                    day_of_year=pl.col("date").dt.ordinal_day(),
                    cos_day_of_year=(pl.col("day_of_year") * 2 * math.pi / 365.0).cos(),
                    month_of_year=pl.col("date").dt.month(),
                    monsoon_season=monsoon_season,
                )
                .filter(
                    pl.col("year") == year,
                )
            )

            self.combined_storage.sink_stage(
                lf_for_year,
                self.output_data_artifact.stage,
            )
