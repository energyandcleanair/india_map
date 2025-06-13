"""Load the grid from a shapefile zip file."""

import tempfile
import zipfile
from pathlib import Path

import geopandas
from geopandas import GeoDataFrame


# The shapefile zip contains the grid for the NED dataset.
# It has a directory structure like this:
# - grid_india_10km/
#   - grid_india_10km.shp
#   - grid_india_10km.shx
#   - grid_india_10km.dbf
#   - grid_india_10km.prj
def load_grid_from_zip(path_to_shapefile_zip: str) -> GeoDataFrame:
    """Load the grid from a file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # first unzip to a temporary directory
        with zipfile.ZipFile(path_to_shapefile_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # construct the path to the shapefile
        shapefile_path = Path(temp_dir, "grid_india_10km")

        # read the shapefile for the grid
        grid = geopandas.read_file(shapefile_path)

        # get the grid cell centroids (in lat/lon)
        centroids = grid.geometry.centroid.to_crs(epsg=4326)

        grid["lon"] = centroids.x
        grid["lat"] = centroids.y

        return grid.to_crs("EPSG:4326")
