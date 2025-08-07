"""Type helpers for providers."""

from typing import Protocol

from pm25ml.training.model_pipeline import ModelPipeline, ModelReference


class ModelTrainerFactory(Protocol):
    """
    Protocol for a factory that creates ModelPipeline instances.

    This factory is used to create model trainers for different models.
    """

    def __call__(
        self,
        model_reference: ModelReference,
    ) -> ModelPipeline:
        """Create a ModelPipeline instance for the given model reference."""
        ...
