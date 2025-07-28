"""Storage for ML models."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import lightgbm
import pandas as pd
from arrow import Arrow
from fsspec import AbstractFileSystem
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from pm25ml.training.compatible_model import Pm25mlCompatibleModel

if TYPE_CHECKING:
    from io import BufferedWriter


@dataclass
class ModelStats:
    """Statistics for a model run."""

    cv_results: pd.DataFrame
    test_metrics: dict


@dataclass
class ValidatedModel(ModelStats):
    """A validated model with its metadata."""

    model: Pm25mlCompatibleModel

    @property
    def type(self) -> str:
        """Return the type of the model."""
        return self.model.__class__.__name__


class Predictor(Protocol):
    """Protocol for a model that can make predictions, this is."""

    def predict(self, data: Any) -> Any:  # noqa: ANN401
        """Make predictions on the provided data."""


@dataclass
class LoadedValidatedModel(ModelStats):
    """
    A validated model loaded from storage.

    This abstracts the type of model away when it's loaded as not all models can be loaded as the
    whole input. For example, the LGBMRegressor can't be loaded fully.
    """

    model: Predictor


type ModelRef = Arrow


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

    def save_model(self, model_name: str, model_run_ref: ModelRef, model: ValidatedModel) -> None:
        """
        Save the validated model to the storage.

        Args:
            model_name (str): The name of the model to save.
            model_run_ref (Arrow): A reference for the model run.
            model (ValidatedModel): The validated model to save.

        """
        base_path = Path(
            self.bucket_name,
            model_name,
            model_run_ref.format("YYYY-MM-DD+HH-mm-ss"),
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
        model: Pm25mlCompatibleModel,
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

    def load_model(self, model_name: str, model_run_ref: ModelRef) -> LoadedValidatedModel:
        """
        Load a validated model from the storage.

        Args:
            model_name (str): The name of the model to load.
            model_run_ref (ModelRef): A reference for the model run, typically a timestamp.

        Returns:
            ValidatedModel: The loaded validated model.

        """
        return self._load_from_str_ref(model_name, model_run_ref.format("YYYY-MM-DD+HH-mm-ss"))

    def _load_from_str_ref(self, model_name: str, model_run_ref: str) -> LoadedValidatedModel:
        base_path = Path(
            self.bucket_name,
            model_name,
            model_run_ref,
        )

        model_type_path = next(iter(self.filesystem.glob(str(base_path / "model+*"))), None)
        if not model_type_path:
            model_type_not_found_msg = "Model type file not found."
            raise FileNotFoundError(model_type_not_found_msg)

        # Ensure `model_type` is defined and accessible
        model_type = model_type_path.split("+")[-1]

        with self.filesystem.open(model_type_path, "rb") as f:
            model_bytes = cast("bytes", f.read())

        predictor: Predictor

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(
                tmp_dir,
                "model.json" if model_type == "XGBRegressor" else "model.txt",
            )

            with model_path.open("wb") as tmp_file:
                tmp_file.write(model_bytes)

            # Ensure proper indentation and fix syntax errors
            if model_type == "XGBRegressor":
                predictor = XGBRegressor()
                predictor.load_model(model_path)
            elif model_type == "LGBMRegressor":
                predictor = lightgbm.Booster(model_file=str(model_path))
            else:
                msg = "Unsupported model type for loading."
                raise TypeError(msg)

        cv_results_path = str(base_path / "cv_results.parquet")
        with self.filesystem.open(cv_results_path, "rb") as f:
            cv_results = pd.read_csv(f)

        test_metrics_path = str(base_path / "test_metrics.json")
        with self.filesystem.open(test_metrics_path, "r") as f:
            test_metrics = json.load(f)

        return LoadedValidatedModel(
            model=predictor,
            cv_results=cv_results,
            test_metrics=test_metrics,
        )

    def load_latest_model(self, model_name: str) -> LoadedValidatedModel:
        """
        Load the latest validated model for a given model name.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            LoadedValidatedModel: The loaded validated model.

        """
        base_path = Path(self.bucket_name, model_name)

        # Find the latest model run reference
        model_run_refs: list[str] = [
            path
            for path in cast("list[str]", self.filesystem.glob(str(base_path / "*")))
            if self.filesystem.isdir(path)
        ]
        if not model_run_refs:
            msg = f"No model runs found for model: {model_name}"
            raise FileNotFoundError(msg)

        latest_run_ref: str = max(model_run_refs)

        # Delegate to the existing load_model method
        return self._load_from_str_ref(model_name, latest_run_ref)
