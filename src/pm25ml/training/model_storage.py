"""Storage for ML models."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from fsspec import AbstractFileSystem
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from pm25ml.training.available_model_types import AvailableModelType

if TYPE_CHECKING:
    from io import BufferedWriter


@dataclass
class ValidatedModel:
    """A validated model with its metadata."""

    model_name: str
    model_run_ref: str
    model: AvailableModelType
    cv_results: pd.DataFrame
    test_metrics: dict

    @property
    def type(self) -> str:
        """Return the type of the model."""
        return self.model.__class__.__name__


class ModelStorage:
    """Storage for ML models."""

    def __init__(self, filesystem: AbstractFileSystem, bucket_name: str) -> None:
        """
        Initialize the model storage.

        Args:
            filesystem (AbstractFileSystem): The filesystem to use for storage.
            bucket_name (str): The name of the bucket where models will be stored.

        """
        self.filesystem = filesystem
        self.bucket_name = bucket_name

    def save_model(self, model: ValidatedModel) -> None:
        """
        Save the validated model to the storage.

        Args:
            model (ValidatedModel): The validated model to save.

        """
        base_path = Path(
            self.bucket_name,
            model.model_name,
            model.model_run_ref,
        )

        self.filesystem.mkdir(str(base_path), create_parents=True, exist_ok=True)

        model_path = str(base_path / f"model+{model.type}")
        with cast("BufferedWriter", self.filesystem.open(model_path, "wb")) as f:
            f.write(self._serialised_to_bytes(model.model))

        cv_results_path = str(base_path / "cv_results.parquet")
        with self.filesystem.open(cv_results_path, "wb") as f:
            model.cv_results.to_csv(f, index=False)

        test_metrics_path = str(base_path / "test_metrics.json")
        with self.filesystem.open(test_metrics_path, "w") as f:
            json.dump(model.test_metrics, f)

    def _serialised_to_bytes(
        self,
        model: AvailableModelType,
    ) -> bytes:
        with tempfile.TemporaryDirectory() as tmp_dir:
            """Save the model to a temporary directory."""
            if isinstance(model, XGBRegressor):
                model_path = Path(tmp_dir, "model.json")
                model.save_model(model_path)
                with model_path.open("rb") as f:
                    return f.read()
            elif isinstance(model, LGBMRegressor):
                model_path = Path(tmp_dir, "model.txt")
                model.booster_.save_model(model_path)
                with model_path.open("rb") as f:
                    return f.read()
            else:
                msg = "Unsupported model type for saving."
                raise TypeError(msg)
