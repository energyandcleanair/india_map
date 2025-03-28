# export_CO.py
import ee
import pandas as pd
from dateutil.relativedelta import relativedelta
from gee_utils import process_image, generate_date_lists, get_export_properties

OUTPUT_DRIVE = "s5p_CO_mean"
START_DATE = "2018-07-01"
END_DATE = "2018-10-01"
# Use monthly increments (step_months=1) for a shorter period
start_list, end_list = generate_date_lists(START_DATE, END_DATE, step_months=1)

collection = ee.ImageCollection(
    "COPERNICUS/S5P/OFFL/L3_CO").select("CO_column_number_density")
export_properties = get_export_properties(["grid_id", "mean", "start_date"])

for s_date, e_date in zip(start_list, end_list):
    fc = process_image(s_date, e_date, collection)
    output_name = "_".join(["CO_10km_grid", s_date, "to", e_date])
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        folder=OUTPUT_DRIVE,
        description=output_name,
        fileFormat="CSV",
        selectors=export_properties
    )
    print("Saving file as", output_name)
    task.start()
