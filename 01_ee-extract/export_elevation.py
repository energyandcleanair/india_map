# export_elevation.py
import ee
from gee_utils import grid, get_export_properties

OUTPUT_DRIVE = "elevation_FIN"

# Load the SRTM elevation image (2000)
elevation = ee.Image("USGS/SRTMGL1_003").select("elevation")
# Create a FeatureCollection by reducing the elevation image over your grid
fc = elevation.reduceRegions(
    collection=grid.limit(41000),
    reducer=ee.Reducer.mean(),
    crs='EPSG:7755',
    scale=10000
).flatten()

export_properties = get_export_properties(["grid_id", "mean"])

output_name = "elevation_10km_2000"
task = ee.batch.Export.table.toDrive(
    collection=fc,
    folder=OUTPUT_DRIVE,
    description=output_name,
    fileFormat="CSV",
    selectors=export_properties
)
print("Saving file as", output_name)
task.start()
