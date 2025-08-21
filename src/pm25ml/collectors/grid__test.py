import numpy as np
import polars as pl

from pm25ml.collectors.grid import Grid
from pm25ml.collectors.geo_time_grid_dataset import DIMS3


def _make_minimal_grid() -> Grid:
    df = pl.DataFrame(
        {
            "grid_id": [1, 2],
            # centroid in original CRS (arbitrary values)
            "original_x": [10.0, 20.0],
            "original_y": [5.0, 5.0],
            # lon/lat (not used by to_xarray_with_data but required by Grid ctor)
            "lon": [77.0, 78.0],
            "lat": [28.0, 28.1],
        }
    )
    return Grid(df)


def test__to_xarray_with_data__happy_path__returns_geo_time_grid_dataset() -> None:
    grid = _make_minimal_grid()

    data_df = pl.DataFrame(
        {
            "grid_id": [1, 2],
            "date": ["2023-01-01", "2023-01-01"],
            "aod": [0.5, 0.7],
            "pm25": [12.3, 45.6],
        }
    )

    ds = grid.to_xarray_with_data(data_df)

    # Dimensions and coordinates
    assert tuple(ds.dims) == DIMS3
    for c in DIMS3:
        assert c in ds.coords

    # time is datetime64
    assert str(ds.coords["time"].dtype).startswith("datetime64")

    # Variable dtypes are float32 (values explicitly cast in implementation)
    assert ds["aod"].dtype == np.float32
    assert ds["pm25"].dtype == np.float32

    # Coordinates reflect original_x/y; our y has a single unique value and x two values
    xs = ds.coords["x"].values
    ys = ds.coords["y"].values
    assert ys.shape == (1,)
    assert xs.shape == (2,)
    np.testing.assert_allclose(xs, [10.0, 20.0], rtol=0, atol=0)
    np.testing.assert_allclose(ys, [5.0], rtol=0, atol=0)

    # Values align to time/y/x grid
    aod_t0 = ds["aod"].isel(time=0).values
    assert aod_t0.shape == (1, 2)
    np.testing.assert_allclose(aod_t0, [[0.5, 0.7]], rtol=1e-6, atol=1e-7)


def test__to_xarray_with_data__missing_id_columns__raises_value_error() -> None:
    grid = _make_minimal_grid()

    # Missing 'date'
    bad_df = pl.DataFrame({"grid_id": [1], "aod": [0.1]})

    try:
        _ = grid.to_xarray_with_data(bad_df)
        raise AssertionError("Expected ValueError for missing 'date' column")
    except ValueError as e:
        assert "Missing ID columns" in str(e)


def test__to_xarray_with_data__parses_date_str__time_coord_datetime64() -> None:
    grid = _make_minimal_grid()
    df = pl.DataFrame(
        {
            "grid_id": [1],
            "date": ["2024-02-29"],  # leap day as a parsing edge
            "aod": [1.23],
        }
    )

    ds = grid.to_xarray_with_data(df)
    assert tuple(ds.dims) == DIMS3
    assert str(ds.coords["time"].dtype).startswith("datetime64")
    # sanity on the parsed date value
    assert np.datetime64("2024-02-29") == ds.coords["time"].values[0]
