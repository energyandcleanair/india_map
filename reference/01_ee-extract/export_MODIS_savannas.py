# export_MODIS_savannas.py
import ee
from gee_utils import process_yearly, grid, get_export_properties

OUTPUT_DRIVE = "MODIS_savannas_FIN"
# Define a list of yearly start and end dates (e.g., from 2005 to 2023)
start_date = "2005-01-01"
end_date = "2023-01-01"

# Build the MODIS savannas product from MCD12Q1.
# (Here, we assume savannas are defined by a certain LC_Type1 value, adjust as needed.)
maiac_old = ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1") \
    .map(lambda img: img.updateMask(
        img.select("LC_Type1").eq(9))) \
    .map(lambda img: img.addBands(
        img.select("LC_Type1")
           .where(img.select("LC_Type1").eq(9), 1)
           .where(img.select("LC_Type1").neq(9), 0)
           .rename("LC_Type1_updated")))
maiac = maiac_old.select("LC_Type1_updated")

export_properties = get_export_properties(["grid_id", "sum", "start_date"])
# Use a sum reducer as in your code
fc = process_yearly(start_date, end_date, maiac, reducer=ee.Reducer.sum())
output_name = "MODIS_savannas_" + start_date + "_to_" + end_date
task = ee.batch.Export.table.toDrive(
    collection=fc,
    folder=OUTPUT_DRIVE,
    description=output_name,
    fileFormat="CSV",
    selectors=export_properties
)
print("Saving file as", output_name)
task.start()
