# export_MODIS_shrub.py
import ee
from gee_utils import process_yearly, grid, get_export_properties

OUTPUT_DRIVE = "MODIS_shrub_FIN"
start_date = "2005-01-01"
end_date = "2023-01-01"

# Define your MODIS shrub product.
# (Adjust the LC_Type1 mask to target shrub areas, e.g., LC_Type1 == 13)
maiac_old = ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1") \
    .map(lambda img: img.updateMask(img.select("LC_Type1").eq(13))) \
    .map(lambda img: img.addBands(
        img.select("LC_Type1")
           .where(img.select("LC_Type1").eq(13), 1)
           .where(img.select("LC_Type1").neq(13), 0)
           .rename("LC_Type1_updated")))
maiac = maiac_old.select("LC_Type1_updated")
export_properties = get_export_properties(["grid_id", "sum", "start_date"])

fc = process_yearly(start_date, end_date, maiac, reducer=ee.Reducer.sum())
output_name = "MODIS_shrub_" + start_date + "_to_" + end_date
task = ee.batch.Export.table.toDrive(
    collection=fc,
    folder=OUTPUT_DRIVE,
    description=output_name,
    fileFormat="CSV",
    selectors=export_properties
)
print("Saving file as", output_name)
task.start()
