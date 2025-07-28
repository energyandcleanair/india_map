import warnings
import pytest
import pandas as pd
import json
from morefs.memory import MemFS
from xgboost import XGBRegressor
from lightgbm import Booster, LGBMRegressor
from pm25ml.training.model_storage import ModelStorage, ValidatedModel


@pytest.fixture
def in_memory_filesystem():
    return MemFS()


@pytest.fixture(scope="module")
def sample_data():
    # Simple dataset for training
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [1, 2, 3, 4]
    return X, y


@pytest.fixture()
def model_storage(in_memory_filesystem):
    return ModelStorage(in_memory_filesystem, "test_bucket")


@pytest.fixture(scope="module")
def trained_xgb_model(sample_data):
    X, y = sample_data
    model = XGBRegressor()
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def trained_lgbm_model(sample_data):
    X, y = sample_data
    model = LGBMRegressor(min_data_in_leaf=1)
    model.fit(X, y)
    return model


def test_save_xgb_model(model_storage, trained_xgb_model):
    validated_model = ValidatedModel(
        model=trained_xgb_model,
        cv_results=pd.DataFrame({"metric": [0.1, 0.2]}),
        test_metrics={"rmse": 0.15},
    )

    model_storage.save_model("xgb_model", "run_1", validated_model)

    assert model_storage.filesystem.exists("test_bucket/xgb_model/run_1/model+XGBRegressor")
    assert model_storage.filesystem.exists("test_bucket/xgb_model/run_1/cv_results.parquet")
    assert model_storage.filesystem.exists("test_bucket/xgb_model/run_1/test_metrics.json")


def test_save_lgbm_model(model_storage, trained_lgbm_model):
    validated_model = ValidatedModel(
        model=trained_lgbm_model,
        cv_results=pd.DataFrame({"metric": [0.3, 0.4]}),
        test_metrics={"mae": 0.25},
    )

    model_storage.save_model("lgbm_model", "run_2", validated_model)

    assert model_storage.filesystem.exists("test_bucket/lgbm_model/run_2/model+LGBMRegressor")
    assert model_storage.filesystem.exists("test_bucket/lgbm_model/run_2/cv_results.parquet")
    assert model_storage.filesystem.exists("test_bucket/lgbm_model/run_2/test_metrics.json")


def test_save_statistics(model_storage, sample_data):
    X, y = sample_data
    model = XGBRegressor()
    model.fit(X, y)

    validated_model = ValidatedModel(
        model=model,
        cv_results=pd.DataFrame({"metric": [0.5, 0.6]}),
        test_metrics={"r2": 0.85},
    )

    model_storage.save_model("stats_model", "run_3", validated_model)

    with model_storage.filesystem.open("test_bucket/stats_model/run_3/cv_results.parquet") as f:
        cv_results = pd.read_csv(f)
        assert "metric" in cv_results.columns

    with model_storage.filesystem.open("test_bucket/stats_model/run_3/test_metrics.json") as f:
        test_metrics = json.load(f)
        assert test_metrics["r2"] == 0.85


def test_load_xgb_model(model_storage, trained_xgb_model):
    validated_model = ValidatedModel(
        model=trained_xgb_model,
        cv_results=pd.DataFrame({"metric": [0.1, 0.2]}),
        test_metrics={"rmse": 0.15},
    )

    model_storage.save_model("xgb_model", "run_1", validated_model)

    loaded_model_wrapper = model_storage.load_model("xgb_model", "run_1")

    loaded_model = loaded_model_wrapper.model
    assert isinstance(loaded_model, XGBRegressor)
    assert loaded_model.get_booster().get_dump() == trained_xgb_model.get_booster().get_dump()
    assert loaded_model.predict([[1, 2]]) == trained_xgb_model.predict([[1, 2]])


def test_load_lgbm_model(model_storage, trained_lgbm_model):
    validated_model = ValidatedModel(
        model=trained_lgbm_model,
        cv_results=pd.DataFrame({"metric": [0.3, 0.4]}),
        test_metrics={"mae": 0.25},
    )

    model_storage.save_model("lgbm_model", "run_2", validated_model)

    loaded_model = model_storage.load_model("lgbm_model", "run_2")

    assert isinstance(loaded_model.model, Booster)
    assert loaded_model.model.dump_model() == trained_lgbm_model.booster_.dump_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert loaded_model.model.predict([[1, 2]]) == trained_lgbm_model.predict([[1, 2]])
