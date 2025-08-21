"""Type support for a geo time grid dataset."""

from typing import Final, NewType, cast

from xarray import Dataset

GeoTimeGridDataset = NewType("GeoTimeGridDataset", Dataset)
"""
Represents an xarray dataset with dimensions (time, y, x) where x and y are in the
original coordinate system.
"""


TIME: Final = "time"
Y: Final = "y"
X: Final = "x"
DIMS3: Final = (TIME, Y, X)


class GeoSchemaError(ValueError):
    """Raised when a dataset fails GeoTimeGrid schema validation."""


def as_geo_time_grid(ds: Dataset) -> GeoTimeGridDataset:
    """
    Convert a Dataset to a GeoTimeGridDataset.

    Checks the dimensions and coordinates of the dataset.
    """
    if tuple(ds.dims) != DIMS3:
        msg = f"Expected dims {DIMS3}, got {tuple(ds.dims)}"
        raise GeoSchemaError(msg)
    for c in DIMS3:
        if c not in ds.coords:
            msg = f"Missing coordinate '{c}'"
            raise GeoSchemaError(msg)
    # dtype sanity
    if not str(ds.coords[TIME].dtype).startswith("datetime64"):
        msg = "time coord must be datetime64"
        raise GeoSchemaError(msg)
    return cast("GeoTimeGridDataset", ds)
