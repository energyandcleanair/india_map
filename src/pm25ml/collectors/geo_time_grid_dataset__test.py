"""Tests for `as_geo_time_grid`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from .geo_time_grid_dataset import GeoSchemaError, as_geo_time_grid


def _make_valid_ds(t: int = 3, y: int = 2, x: int = 4) -> xr.Dataset:
    """Helper: build a minimal valid dataset with dims (time, y, x)."""
    time = np.array([np.datetime64("2023-01-01") + np.timedelta64(i, "D") for i in range(t)])
    yy = np.arange(y, dtype=np.int32)
    xx = np.arange(x, dtype=np.int32)
    data = np.zeros((t, y, x), dtype=np.float32)
    return xr.Dataset(
        data_vars={"a": (("time", "y", "x"), data)},
        coords={"time": time, "y": yy, "x": xx},
    )


def test__as_geo_time_grid__valid_dataset__returns_input() -> None:
    ds = _make_valid_ds()
    out = as_geo_time_grid(ds)
    # NewType cast should return the same object at runtime
    assert out is ds


def test__as_geo_time_grid__extra_dim_present__raises_GeoSchemaError() -> None:
    ds = _make_valid_ds()
    # Add an extra dimension to violate the (time, y, x) schema
    band = np.arange(2, dtype=np.int32)
    data4 = np.zeros((ds.sizes["time"], ds.sizes["y"], ds.sizes["x"], 2), dtype=np.float32)
    ds_bad = xr.Dataset(
        data_vars={"a": (("time", "y", "x", "band"), data4)},
        coords={
            "time": ds.coords["time"].values,
            "y": ds.coords["y"].values,
            "x": ds.coords["x"].values,
            "band": band,
        },
    )
    with pytest.raises(GeoSchemaError):
        as_geo_time_grid(ds_bad)


def test__as_geo_time_grid__time_coord_not_datetime64__raises_GeoSchemaError() -> None:
    t, y, x = 3, 2, 4
    # Use integer time coordinate (invalid)
    time = np.arange(t, dtype=np.int32)
    yy = np.arange(y, dtype=np.int32)
    xx = np.arange(x, dtype=np.int32)
    data = np.zeros((t, y, x), dtype=np.float32)
    ds_bad = xr.Dataset(
        data_vars={"a": (("time", "y", "x"), data)},
        coords={"time": time, "y": yy, "x": xx},
    )
    with pytest.raises(GeoSchemaError):
        as_geo_time_grid(ds_bad)


def test__as_geo_time_grid__missing_coordinate__raises_GeoSchemaError() -> None:
    # Build a dataset that is missing the 'y' coordinate entry
    t, y, x = 2, 2, 3
    time = np.array([np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(t)])
    yy = np.arange(y, dtype=np.int32)
    xx = np.arange(x, dtype=np.int32)
    data = np.zeros((t, y, x), dtype=np.float32)
    ds_bad = xr.Dataset(
        data_vars={"a": (("time", "y", "x"), data)},
        coords={"time": time, "x": xx, "y": ("y", yy)},
    )
    # Deliberately drop the 'y' coordinate key
    ds_bad = ds_bad.drop_vars("y")
    with pytest.raises(GeoSchemaError):
        as_geo_time_grid(ds_bad)
