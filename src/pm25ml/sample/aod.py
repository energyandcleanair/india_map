"""Sampling for AOD."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from dependency_injector.wiring import Provide, inject

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.feature_generation.generate import GENERATED_FEATURES_STAGE
from pm25ml.hive_path import HivePath
from pm25ml.logging import logger
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env


def _process_month(month: str, combined_storage: CombinedStorage) -> None:
    logger.info(f"Sampling for AOD for month: {month}")
    monthly_data = combined_storage.read_dataframe(
        result_subpath=HivePath.from_args(
            stage=GENERATED_FEATURES_STAGE,
            month=month,
        ),
        file_name="0.parquet",
    ).filter(
        pl.col("modis_aod__Optical_Depth_055").is_not_null(),
    )

    sampled_keys = (
        monthly_data.select(
            "grid_id",
            "date",
            "grid__id_50km",
        )
        .group_by("grid__id_50km")
        .map_groups(
            lambda df: df.sample(fraction=0.03, with_replacement=False, seed=42),
        )
        .select(
            "grid_id",
            "date",
        )
        .with_columns(
            pl.lit("training").alias("split"),
        )
    )

    split_dataset = monthly_data.join(
        sampled_keys,
        on=["grid_id", "date"],
        how="left",
        coalesce=True,
    ).with_columns(
        split=pl.col("split").fill_null("test"),
    )

    combined_storage.write_to_destination(
        split_dataset,
        HivePath.from_args(
            stage="sampled",
            model="aod",
            month=month,
        ),
    )


@inject
def _main(
    combined_storage: CombinedStorage = Provide[Pm25mlContainer.combined_storage],
    temporal_config: TemporalConfig = Provide[Pm25mlContainer.temporal_config],
) -> None:
    months = temporal_config.month_ids

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process_month, month, combined_storage) for month in months]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
