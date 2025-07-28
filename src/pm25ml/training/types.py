"""Defines the available models for imputation."""

from typing import Literal

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

type Pm25mlCompatibleModel = XGBRegressor | LGBMRegressor

type ModelName = Literal["aod"]
