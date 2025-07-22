"""Feature generation for PM2.5 data."""

import math

import polars as pl
from dependency_injector.wiring import Provide, inject

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.recombiner.recombiner import Recombiner
from pm25ml.logging import logger
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env

ABSOLUTE_ZERO = 273.15
MAGNUS_APPROXIMATION_A = 17.625
MAGNUS_APPROXIMATION_B = 234.04
MONSOON_SEASON_MONTHS = [6, 7, 8, 9]  # June to September


@inject
def _main(
    combined_storage: CombinedStorage = Provide[Pm25mlContainer.combined_storage],
    recombiner: Recombiner = Provide[Pm25mlContainer.spatial_interpolation_recombiner],
    temporal_config: TemporalConfig = Provide[Pm25mlContainer.temporal_config],
) -> None:
    # Test writing random file
    test_table = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    combined_storage.write_to_destination(test_table, "test/random")

    lf = combined_storage.scan_stage(recombiner.new_stage_name)
    for year in temporal_config.years:
        logger.info(f"Generating features for year: {year}")

        # This window must include the current year and the previous year
        months_in_window = [
            month.format("YYYY-MM")
            for month in temporal_config.months
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
            return pl.col(col_name).rolling_mean(7, min_samples=1).over("grid_id")

        def annual_rolling_mean(col_name: str) -> pl.Expr:
            return pl.col(col_name).rolling_mean(365, min_samples=1).over("grid_id")

        def annual_average(col_name: str) -> pl.Expr:
            return pl.col(col_name).mean().over(["grid_id", "year"])

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
            .filter((pl.col("year") == year) | (pl.col("year") == year - 1))
            .with_columns(
                era5_land__c_relative_humidity=relative_humidity,
                era5_land__c_wind_degree=wind_degree,
            )
            .with_columns(
                merra_aot__aot__7d=weekly_rolling_mean("merra_aot__aot"),
                merra_aot__aot__365d=annual_rolling_mean("merra_aot__aot"),
                merra_co__co__7d=weekly_rolling_mean("merra_co__co"),
                merra_co__co__365d=annual_rolling_mean("merra_co__co"),
                merra_co__co__grid_year_mean=annual_average("merra_co__co"),
                merra_co_top__co__7d=weekly_rolling_mean("merra_co_top__co"),
                merra_co_top__co__365d=annual_rolling_mean("merra_co_top__co"),
                merra_co_top__co__grid_year_mean=annual_average("merra_co_top__co"),
                era5_land__temperature_2m__7d=weekly_rolling_mean("era5_land__temperature_2m"),
                era5_land__temperature_2m__grid_year_mean=annual_average(
                    "era5_land__temperature_2m",
                ),
                era5_land__dewpoint_temperature_2m__7d=weekly_rolling_mean(
                    "era5_land__dewpoint_temperature_2m",
                ),
                era5_land__dewpoint_temperature_2m__grid_year_mean=annual_average(
                    "era5_land__dewpoint_temperature_2m",
                ),
                era5_land__c_relative_humidity__7d=weekly_rolling_mean(
                    "era5_land__c_relative_humidity",
                ),
                era5_land__c_relative_humidity__365d=annual_rolling_mean(
                    "era5_land__c_relative_humidity",
                ),
                era5_land__c_wind_degree__7d=weekly_rolling_mean("era5_land__c_wind_degree"),
                era5_land__c_wind_degree__365d=annual_rolling_mean("era5_land__c_wind_degree"),
                era5_land__u_component_of_wind_10m__7d=weekly_rolling_mean(
                    "era5_land__u_component_of_wind_10m",
                ),
                era5_land__u_component_of_wind_10m__365d=annual_rolling_mean(
                    "era5_land__u_component_of_wind_10m",
                ),
                era5_land__v_component_of_wind_10m__7d=weekly_rolling_mean(
                    "era5_land__v_component_of_wind_10m",
                ),
                era5_land__v_component_of_wind_10m__365d=annual_rolling_mean(
                    "era5_land__v_component_of_wind_10m",
                ),
                era5_land__total_precipitation_sum__7d=weekly_rolling_mean(
                    "era5_land__total_precipitation_sum",
                ),
                era5_land__total_precipitation_sum__365d=annual_rolling_mean(
                    "era5_land__total_precipitation_sum",
                ),
                era5_land__surface_net_thermal_radiation_sum__7d=weekly_rolling_mean(
                    "era5_land__surface_net_thermal_radiation_sum",
                ),
                era5_land__surface_net_thermal_radiation_sum__365d=annual_rolling_mean(
                    "era5_land__surface_net_thermal_radiation_sum",
                ),
                era5_land__surface_pressure__7d=weekly_rolling_mean("era5_land__surface_pressure"),
                era5_land__surface_pressure__365d=annual_rolling_mean(
                    "era5_land__surface_pressure",
                ),
                day_of_year=pl.col("date").dt.ordinal_day(),
                monsoon_season=monsoon_season,
            )
            .filter(
                pl.col("year") == year,
            )
        )

        combined_storage.sink_stage(
            lf_for_year,
            "generated_features",
        )


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
