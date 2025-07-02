from polars import Float64, Int64, String
from pathlib import Path

import pytest
from pm25ml.collectors.grid_loader import load_grid_from_zip, Grid

from shapely.wkt import loads as load_wkt
from shapely.geometry import Polygon
from typing import cast

pytestmark = pytest.mark.integration

SAMPLE_GRID_ID = 61788
GRID_ID_7317_EXPECTED_POLYGON = (
    "POLYGON (("
    "92.91545192768046 23.55959233529598, "
    "92.92460218988784 23.65128722473793, "
    "93.02413560355933 23.642819806575076, "
    "93.01491567514682 23.551130887157257, "
    "92.91545192768046 23.55959233529598))"
)
GRID_ID_7317_EXPECTED_CENTROID_LON = 92.96977688113475
GRID_ID_7317_EXPECTED_CENTROID_LAT = 23.601212914119746

EXPECTED_N_ROWS = 33074

MAX_ERROR_SIZE = 1e-10


def test__load_grid_from_zip__valid_shapefile_zip__grid_loaded():
    """Test loading a grid from a valid shapefile zip file."""
    # Arrange
    path_to_shapefile_zip = Path("grid_india_10km_shapefiles.zip")

    # Act
    grid = load_grid_from_zip(path_to_shapefile_zip)

    assert isinstance(grid, Grid)

    # Assert the shape and types of the data frame
    assert grid.df.shape[0] == EXPECTED_N_ROWS
    assert "geometry_wkt" in grid.df.columns
    assert grid.df["geometry_wkt"].dtype == String
    assert "lon" in grid.df.columns
    assert grid.df["lon"].dtype == Float64
    assert "lat" in grid.df.columns
    assert grid.df["lat"].dtype == Float64
    assert "grid_id" in grid.df.columns
    assert grid.df["grid_id"].dtype == Int64

    # Make sure that all of the lat and lon values are within expected ranges
    lon_min = grid.df["lon"].min()
    lon_max = grid.df["lon"].max()
    assert isinstance(lon_min, (float, int)) and lon_min >= -180
    assert isinstance(lon_max, (float, int)) and lon_max <= 180
    lat_min = grid.df["lat"].min()
    lat_max = grid.df["lat"].max()
    assert isinstance(lat_min, (float, int)) and lat_min >= -90
    assert isinstance(lat_max, (float, int)) and lat_max <= 90

    # Check that the sampled GRID ID has a polygon close to the expected one
    sample_row = grid.df.filter(grid.df["grid_id"] == SAMPLE_GRID_ID)
    polygon_wkt = sample_row["geometry_wkt"].item()
    expected_wkt: Polygon = cast(Polygon, load_wkt(GRID_ID_7317_EXPECTED_POLYGON))
    actual_wkt = load_wkt(polygon_wkt)
    assert isinstance(actual_wkt, Polygon), "Sample geometry is not a Polygon"
    assert len(list(expected_wkt.exterior.coords)) == len(list(actual_wkt.exterior.coords)), (
        "Number of points in polygon does not match"
    )
    for expected_coord, actual_coord in zip(
        expected_wkt.exterior.coords, actual_wkt.exterior.coords
    ):
        assert abs(expected_coord[0] - actual_coord[0]) < MAX_ERROR_SIZE, (
            "Longitude coordinates do not match on shape"
        )
        assert abs(expected_coord[1] - actual_coord[1]) < MAX_ERROR_SIZE, (
            "Latitude coordinates do not match on shape"
        )

    # Check that the sampled GRID ID has a centroid close to the expected one
    centroid = actual_wkt.centroid
    assert abs(centroid.x - GRID_ID_7317_EXPECTED_CENTROID_LON) < MAX_ERROR_SIZE, (
        "Centroid longitude does not match"
    )
    assert abs(centroid.y - GRID_ID_7317_EXPECTED_CENTROID_LAT) < MAX_ERROR_SIZE, (
        "Centroid latitude does not match"
    )
