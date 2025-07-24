"""Configuration and definition of samplers for PM2.5 ML project."""

from collections.abc import Collection
from dataclasses import dataclass

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.sample.imputation_sampler import (
    ImputationSamplerDefinition,
    SpatialTemporalImputationSampler,
)
from pm25ml.setup.date_params import TemporalConfig


@dataclass
class ImputationStep:
    """Configuration for imputation steps."""

    imputation_sampler_definition: ImputationSamplerDefinition


def define_samplers(
    combined_storage: CombinedStorage,
    temporal_config: TemporalConfig,
    imputation_steps: Collection[ImputationStep],
) -> Collection[SpatialTemporalImputationSampler]:
    """Define samplers for the PM2.5 ML project."""
    return [
        SpatialTemporalImputationSampler(
            combined_storage=combined_storage,
            temporal_config=temporal_config,
            imputation_sampler_definition=step.imputation_sampler_definition,
        )
        for step in imputation_steps
    ]
