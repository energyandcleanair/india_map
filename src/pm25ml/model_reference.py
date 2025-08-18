"""Reference classes for a variety of model types."""

from abc import ABC
from dataclasses import dataclass
from typing import Callable

import polars as pl

from pm25ml.training.types import ModelName, Pm25mlCompatibleModel


@dataclass
class ModelReference(ABC):
    """
    Provides basic reference information for a model.

    This provides a base class for all model references and includes the required
    information to be able to predict (but not necessarily to train) using a model.
    """

    model_name: ModelName

    predictor_cols: list[str]
    target_col: str

    min_r2_score: float
    max_r2_score: float


@dataclass
class FullModelReference(ModelReference):
    """Data definition for the full model."""

    predictor_cols: list[str]
    target_col: str
    stratifier_col: str
    grouper_col: str

    model_builder: Callable[[], Pm25mlCompatibleModel]

    extra_sampler: Callable[[pl.LazyFrame], pl.LazyFrame]

    min_r2_score: float
    max_r2_score: float

    def __post_init__(self) -> None:
        """Validate the model reference."""
        if self.target_col in self.predictor_cols:
            msg = (
                f"Target column '{self.target_col}' cannot be in predictor columns: "
                f"{self.predictor_cols}"
            )
            raise ValueError(
                msg,
            )
        if self.grouper_col in self.predictor_cols:
            msg = (
                f"Grouper column '{self.grouper_col}' cannot be in predictor columns: "
                f"{self.predictor_cols}"
            )
            raise ValueError(
                msg,
            )
        if self.stratifier_col in self.predictor_cols:
            msg = (
                f"Stratified column '{self.stratifier_col}' cannot be in predictor columns: "
                f"{self.predictor_cols}"
            )
            raise ValueError(
                msg,
            )

    @property
    def all_cols(self) -> set[str]:
        """Return all required columns for the model."""
        return {self.target_col, self.grouper_col, self.stratifier_col, *self.predictor_cols}


@dataclass
class ImputationModelReference(ModelReference):
    """Data definition for the imputation model."""

    model_name: ModelName
    predictor_cols: list[str]
    target_col: str
    grouper_col: str

    model_builder: Callable[[], Pm25mlCompatibleModel]

    extra_sampler: Callable[[pl.LazyFrame], pl.LazyFrame]

    min_r2_score: float
    max_r2_score: float

    def __post_init__(self) -> None:
        """Validate the model reference."""
        if self.target_col in self.predictor_cols:
            msg = (
                f"Target column '{self.target_col}' cannot be in predictor columns: "
                f"{self.predictor_cols}"
            )
            raise ValueError(
                msg,
            )
        if self.grouper_col in self.predictor_cols:
            msg = (
                f"Grouper column '{self.grouper_col}' cannot be in predictor columns: "
                f"{self.predictor_cols}"
            )
            raise ValueError(
                msg,
            )

    @property
    def all_cols(self) -> set[str]:
        """Return all required columns for the model."""
        return {self.target_col, self.grouper_col, *self.predictor_cols}
