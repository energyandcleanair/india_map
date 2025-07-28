"""Defines the available models for imputation."""

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

type Pm25mlCompatibleModel = XGBRegressor | LGBMRegressor
