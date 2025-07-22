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

        def all_averages(col_name: str) -> dict[str, pl.Expr]:
            return {
                f"{col_name}__mean_r7d": weekly_rolling_mean(col_name),
                f"{col_name}__mean_r365d": annual_rolling_mean(col_name),
                f"{col_name}__mean_year": annual_average(col_name),
                f"{col_name}__mean_all": pl.col(col_name).mean().over("grid_id"),
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
                era5_land__c_wind_degree_computed=wind_degree,
            )
            .with_columns(
                **all_averages("merra_aot__aot"),
                **all_averages("merra_co__co"),
                **all_averages("merra_co_top__co"),
                **all_averages("era5_land__temperature_2m"),
                **all_averages("era5_land__dewpoint_temperature_2m"),
                **all_averages("era5_land__relative_humidity_computed"),
                **all_averages("era5_land__c_wind_degree_computed"),
                **all_averages("era5_land__u_component_of_wind_10m"),
                **all_averages("era5_land__v_component_of_wind_10m"),
                **all_averages("era5_land__total_precipitation_sum"),
                **all_averages("era5_land__surface_net_thermal_radiation_sum"),
                **all_averages("era5_land__surface_pressure"),
                **all_averages("omi_no2__no2"),
                day_of_year=pl.col("date").dt.ordinal_day(),
                cos_day_of_year=(pl.col("day_of_year") * 2 * math.pi / 365.0).cos(),
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
