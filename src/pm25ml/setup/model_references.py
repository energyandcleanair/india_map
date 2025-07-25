"""Builds the model references for training data."""

from typing import Callable, Literal

import polars as pl
from xgboost import XGBRegressor

from pm25ml.training.model_pipeline import ModelReference

type AvaiableRef = Literal["aod",]


def build_training_ref(
    ref: AvaiableRef,
    extra_sampler: Callable[[pl.LazyFrame], pl.LazyFrame],
) -> ModelReference:
    """Build a training data reference for the given ref."""
    if ref == "aod":
        return ModelReference(
            source_stage="aod",
            predictor_cols=[
                "merra_aot__aot",
                "merra_co_top__co",
                "merra_co__co",
                "era5_land__v_component_of_wind_10m",
                "era5_land__u_component_of_wind_10m",
                "era5_land__total_precipitation_sum",
                "era5_land__temperature_2m",
                "era5_land__surface_pressure",
                "era5_land__surface_net_thermal_radiation_sum",
                "era5_land__leaf_area_index_low_vegetation",
                "era5_land__leaf_area_index_high_vegetation",
                "era5_land__dewpoint_temperature_2m",
                "modis_aod__Optical_Depth_055",
                "srtm_elevation__elevation",
                "modis_land_cover__water",
                "modis_land_cover__shrub",
                "modis_land_cover__urban",
                "modis_land_cover__forest",
                "modis_land_cover__savanna",
                "month_of_year",
                "day_of_year",
                "cos_day_of_year",
                "monsoon_season",
                "grid__lon",
                "grid__lat",
                "era5_land__wind_degree_computed",
                "era5_land__relative_humidity_computed",
                "merra_aot__aot__mean_r7d",
                "merra_co_top__co__mean_r7d",
                "omi_no2__no2__mean_r7d",
                "era5_land__v_component_of_wind_10m__mean_r7d",
                "era5_land__u_component_of_wind_10m__mean_r7d",
                "era5_land__total_precipitation_sum__mean_r7d",
                "era5_land__temperature_2m__mean_r7d",
                "era5_land__wind_degree_computed__mean_r7d",
                "era5_land__relative_humidity_computed__mean_r7d",
                "era5_land__surface_net_thermal_radiation_sum__mean_r7d",
                "era5_land__dewpoint_temperature_2m__mean_r7d",
                "merra_aot__aot__mean_year",
                "merra_co_top__co__mean_year",
                "omi_no2__no2__mean_year",
                "era5_land__v_component_of_wind_10m__mean_year",
                "era5_land__u_component_of_wind_10m__mean_year",
                "era5_land__total_precipitation_sum__mean_year",
                "era5_land__surface_net_thermal_radiation_sum__mean_year",
                "era5_land__leaf_area_index_low_vegetation__mean_year",
                "era5_land__leaf_area_index_high_vegetation__mean_year",
                "era5_land__dewpoint_temperature_2m__mean_year",
                "era5_land__wind_degree_computed__mean_year",
                "era5_land__relative_humidity_computed__mean_year",
                "merra_co_top__co__mean_all",
            ],
            target_col="modis_aod__Optical_Depth_055",
            grouper_col="grid__id_50km",
            model_builder=lambda: XGBRegressor(
                **{
                    "eta": 0.1,
                    "gamma": 0.8,
                    "max_depth": 20,
                    "min_child_weight": 1,
                    "subsample": 0.8,
                    "lambda": 100,
                    "n_estimators": 1000,
                    "booster": "gbtree",
                },
            ),
            extra_sampler=extra_sampler,
        )

    msg = f"Unknown model reference: {ref}"
    raise ValueError(msg)
