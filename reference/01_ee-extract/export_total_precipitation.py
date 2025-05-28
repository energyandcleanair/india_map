# export_total_precipitation.py
import ee
import pandas as pd
from dateutil.relativedelta import relativedelta
from gee_utils import process_image, generate_date_lists, get_export_properties

OUTPUT_DRIVE = "total_precipitation_FIN"
START_DATE = "2005-01-01"
END_DATE = "2023-10-01"
start_list, end_list = generate_date_lists(START_DATE, END_DATE, step_months=3)

collection = ee.ImageCollection(
    "ECMWF/ERA5_LAND/DAILY_AGGR").select("total_precipitation_sum")
export_properties = get_export_properties(["grid_id", "mean", "start_date"])

for s_date, e_date in zip(start_list, end_list):
    fc = process_image(s_date, e_date, collection)
    output_name = "_".join(["total_precipitation_sum", s_date, "to", e_date])
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        folder=OUTPUT_DRIVE,
        description=output_name,
        fileFormat="CSV",
        selectors=export_properties
    )
    print("Saving file as", output_name)
    task.start()
