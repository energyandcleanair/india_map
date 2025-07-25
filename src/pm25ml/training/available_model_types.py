"""Defines the available models for imputation."""

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

type AvailableModelType = XGBRegressor | LGBMRegressor
