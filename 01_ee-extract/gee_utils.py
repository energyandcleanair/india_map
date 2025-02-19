# gee_utils.py
import ee
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Initialize Earth Engine (run ee.Authenticate() manually the first time)
ee.Initialize()

# Define your grid (update asset path as needed)
# You need to upload grid_india_10km in the google earth engine first before run this script
GRID_ASSET = "path_to_your_asset_in_gogole_earth_engine/grid_india_10km" # collect input features for these grid polygons 
grid = ee.FeatureCollection(GRID_ASSET)

def generate_date_lists(start_str, end_str, step_months=3):
    """
    Generate two lists of date strings.
    The first list is for start dates and the second for end dates,
    with increments of `step_months`.
    """
    start_date = pd.to_datetime(start_str)
    end_date = pd.to_datetime(end_str)
    start_list = []
    while start_date < end_date:
        start_list.append(start_date.strftime("%Y-%m-%d"))
        start_date += relativedelta(months=step_months)
    # For the end dates, start one increment ahead
    start_date = pd.to_datetime(start_str) + relativedelta(months=step_months)
    end_list = []
    while start_date <= end_date:
        end_list.append(start_date.strftime("%Y-%m-%d"))
        start_date += relativedelta(months=step_months)
    return start_list, end_list

def calculate_length(start_date, end_date):
    """Return the number of days between start_date and end_date."""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return (end_date - start_date).days

def create_image(start_date, length_date, collection):
    """Return an ImageCollection of daily composites from the given collection."""
    images_list = ee.List([])
    start_dt = pd.to_datetime(start_date)
    for i in range(length_date):
        im_date = ee.Date.fromYMD(start_dt.year, start_dt.month, start_dt.day).advance(i, 'day')
        im = collection.filterDate(im_date, im_date.advance(1, 'day')).reduce(ee.Reducer.mean())
        im = ee.Image(im).set("start_date", im_date.format("yMMdd"))
        images_list = images_list.add(im)
    return ee.ImageCollection.fromImages(images_list)

def process_image(start_date, end_date, collection, reducer=ee.Reducer.mean(), scale=10000, crs='EPSG:7755'):
    """
    Process the given ImageCollection (daily composites) over the date range.
    Returns a FeatureCollection (one feature per grid cell per day).
    """
    length_date = calculate_length(start_date, end_date)
    images = create_image(start_date, length_date, collection)
    processed = images.map(lambda im: im.reduceRegions(
        collection=grid.limit(41000),
        reducer=reducer,
        crs=crs,
        scale=scale
    ).map(lambda f: f.set("start_date", im.get("start_date"))))
    return ee.FeatureCollection(processed).flatten()

def process_yearly(start_date, end_date, collection, reducer=ee.Reducer.mean(), scale=10000, crs='EPSG:7755'):
    """
    Special routine for yearly products.
    Assumes that the time step is one year.
    """
    # Calculate length in years:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    years = end_date.year - start_date.year
    images_list = ee.List([])
    for i in range(years):
        im_date = ee.Date.fromYMD(start_date.year + i, start_date.month, start_date.day)
        im = collection.filterDate(im_date, im_date.advance(1, 'year')).reduce(ee.Reducer.mean())
        im = ee.Image(im).set("start_date", im_date.format("yMMdd"))
        images_list = images_list.add(im)
    images = ee.ImageCollection.fromImages(images_list)
    processed = images.map(lambda im: im.reduceRegions(
        collection=grid.limit(41000),
        reducer=reducer,
        crs=crs,
        scale=scale
    ).set("start_date", im.get("system:time_start")))
    return ee.FeatureCollection(processed).flatten()

def get_export_properties(fields):
    """Return the export properties (fields) for a given product."""
    return fields
