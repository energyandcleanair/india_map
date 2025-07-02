"""Load the grid from a shapefile zip file."""

import tempfile
import zipfile
from pathlib import Path

import shapefile
from polars import DataFrame
from pyproj import CRS, Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.wkt import loads as load_wkt

from pm25ml.collectors.ned.coord_types import Lat, Lon


class Grid:
    """A class representing a grid for the NED dataset."""

    GEOM_COL = "geometry_wkt"
    LAT_COL = "lat"
    LON_COL = "lon"
    GRID_ID_COL = "grid_id"

    def __init__(self, df: DataFrame) -> None:
        """
        Initialize the Grid with a DataFrame.

        Args:
            df (DataFrame): The DataFrame containing grid data.

        """
        self.df = df

    @property
    def bounds(self) -> tuple[Lon, Lat, Lon, Lat]:
        """Get the bounds of the grid."""
        bounds = [load_wkt(wkt).bounds for wkt in self.df[self.GEOM_COL]]
        minx = min(b[0] for b in bounds)
        miny = min(b[1] for b in bounds)
        maxx = max(b[2] for b in bounds)
        maxy = max(b[3] for b in bounds)
        return (Lon(minx), Lat(miny), Lon(maxx), Lat(maxy))

    @property
    def n_rows(self) -> int:
        """Get the number of rows in the grid."""
        return self.df.shape[0]


# The shapefile zip contains the grid for the NED dataset.
# It has a directory structure like this:
# - grid_india_10km/
#   - grid_india_10km.shp
#   - grid_india_10km.shx
#   - grid_india_10km.dbf
#   - grid_india_10km.prj
def load_grid_from_zip(path_to_shapefile_zip: Path) -> Grid:
    """Load the grid from a file."""
    # Extract ZIP to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(path_to_shapefile_zip, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        tmpdir_path = Path(tmpdir)

        # Find .shp and .prj files recursively
        shp_path = next(tmpdir_path.rglob("*.shp"), None)
        prj_path = next(tmpdir_path.rglob("*.prj"), None)

        if not shp_path:
            msg = "Shapefile (.shp) not found in the ZIP archive."
            raise ValueError(msg)

        if not prj_path:
            msg = "Projection file (.prj) not found in the ZIP archive."
            raise ValueError(msg)

        # Load CRS
        with prj_path.open() as f:
            wkt = f.read()
        input_crs = CRS.from_wkt(wkt)

        output_crs = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)

        def reproject_geom(geom: BaseGeometry) -> BaseGeometry:
            return transform(transformer.transform, geom)

        # Read shapefile
        reader = shapefile.Reader(str(shp_path))
        fields = [f[0] for f in reader.fields[1:]]  # skip deletion flag

        records = []
        for sr in reader.shapeRecords():
            attrs = dict(zip(fields, sr.record))
            # Convert grid_id to int if present
            if "grid_id" not in attrs:
                msg = "grid_id not found in shapefile attributes."
                raise ValueError(msg)

            attrs[Grid.GRID_ID_COL] = int(attrs["grid_id"])
            geom = shape(sr.shape.__geo_interface__)
            geom_reproj = reproject_geom(geom)
            attrs[Grid.GEOM_COL] = geom_reproj.wkt

            # Extract centroid coordinates for lon and lat
            centroid = geom_reproj.centroid
            attrs[Grid.LON_COL] = centroid.x
            attrs[Grid.LAT_COL] = centroid.y

            records.append(attrs)

        # Load into polars
        return Grid(DataFrame(records))
