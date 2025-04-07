import os
from typing import List
import earthaccess
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd


def main(dataset: str, varname: str, var_label: str, outfilename: str,
         dataset_version: str = None,
         start_date: str = "2020-01-01", end_date: str = "2020-01-02",
         temp_data_dir: str = "./temp_data"):

    # read the grid file to interpolate to
    grid = get_grid()

    # get grid bounds
    grid_bounds = grid.total_bounds

    # Authenticate with Earthdata Login
    # create account at https://urs.earthdata.nasa.gov/
    earthaccess.login(persist=True)

    # find files for this time period
    data_available = search_files(
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        version=dataset_version
    )

    # loop over the files
    for granule in data_available:

        filename = granule['meta']['native-id'].split(':')[-1]
        datestr = filename.split('.')[-2]

        # 0) check if the file already exists
        outfile = os.path.join(temp_data_dir, f"{outfilename}_{datestr}.csv")

        if os.path.exists(outfile):
            print(
                f"File {outfile} already exists, skipping download & gridding.")
            continue

        # 1) download file, if it is not on disk yet
        if os.path.exists(os.path.join(temp_data_dir, filename)):
            file = [os.path.join(temp_data_dir, filename)]
        else:
            file = earthaccess.download(granule, temp_data_dir)

        # 2) read data from the file
        # check if the file is a netcdf or hdf5 file and read the data accordingly
        if filename.endswith('.nc4'):
            da, datestr = read_nc(file=file,
                                  var_name=varname,
                                  grid_bounds=grid_bounds)
        elif filename.endswith('.h5') or filename.endswith('.he5'):
            da, datestr = read_h5(file=file,
                                  var_name=varname,
                                  grid_bounds=grid_bounds)

        # 3) interpolate the variable to the grid
        df = interp_to_grid(da=da,
                            grid=grid,
                            var_name=varname,
                            datestr=datestr)

        # 4) save the interpolated data
        data_out = df[['grid_id', 'date', varname]].rename(
            columns={varname: var_label})

        data_out.to_csv(outfile, index=False)

        # 5) clean up the downloaded file
        # os.remove(file)


def get_grid():
    """Read the grid file to interpolate to

    Returns
    -------
    grid : geopandas.GeoDataFrame
        The grid file including polygons for the grid cells
        and the with lat/lon coordinates of the grid cell centroids
    """

    # read the shapefile for the grid
    shapefile = "../../input_data/grid_india_10km"
    grid = gpd.read_file(shapefile)

    # get the grid cell centroids (in lat/lon)
    centroids = grid.geometry.centroid.to_crs(epsg=4326)

    grid['lon'] = centroids.x
    grid['lat'] = centroids.y

    grid = grid.to_crs("EPSG:4326")

    return grid


def search_files(dataset: str, start_date: str, end_date: str, version=None):

    # Search for data files
    # no use in defining a bounding box, all data that we need
    # comes in global files
    results = earthaccess.search_data(
        short_name=dataset,
        temporal=(start_date, end_date),
        count=-1,  # get all records
        version=version  # '003'
    )

    # check if results is empty
    if len(results) == 0:
        raise RuntimeError(
            f"No data found for dataset {dataset} between {start_date} and {end_date}")

    # check that the number of files matches the number of days
    # in the date range
    # TODO this does not work when the last days requested are not yet available
    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    if len(results) != num_days:
        raise ValueError(
            f"Number of files ({len(results)}) does not match number of days ({num_days})"
        )

    return results


def read_nc(file, var_name, grid_bounds):
    """Read the data from the file

    Parameters
    ----------
    file : str
        The path to the file
    var_name : str
        The variable name in the file

    Returns
    -------
    xarray.DataArray
        The data array containing the variable
    str
        The date string from the file
    """

    # if file is a list of length 1, take the first element
    if isinstance(file, list) and len(file) == 1:
        file = file[0]

    # open the file
    with xr.open_dataset(file) as ds:

        # check if the variable is in the dataset
        if var_name not in ds.data_vars:
            raise ValueError(f"Variable {var_name} not found in dataset")

        # crop data to area of interest (with some buffer), to save memory
        ds = ds.sel(lon=slice(grid_bounds[0]-1, grid_bounds[2]+1),
                    lat=slice(grid_bounds[1]-1, grid_bounds[3]+1))

        # if lev is in the dimensions, take the lowest level
        if 'lev' in ds.dims:
            print("Data has levels, taking the values on the lowest level")
            # if positive is down, the lowest level is the highest index
            if ds.lev.attrs['positive'] == 'down':
                ds = ds.isel(lev=-1)
            else:
                ds = ds.isel(lev=0)

        # take time average over the day
        ds = ds.mean(dim='time', keep_attrs=True)

        # check that data is now 2D
        if len(ds.dims) != 2:
            raise ValueError(
                f"Data is not 2D for projection: dimensions are {ds.dims}")

    return ds[var_name], ds.attrs['RangeBeginningDate']


def read_h5(file, var_name, grid_bounds):

    # if file is a list of length 1, take the first element
    if isinstance(file, list) and len(file) == 1:
        file = file[0]

    if var_name in ['ColumnAmountNO2', 'ColumnAmountNO2CloudScreened', 'ColumnAmountNO2Trop', 'ColumnAmountNO2TropCloudScreened']:

        ds = xr.open_dataset(
            file, group='HDFEOS/GRIDS/ColumnAmountNO2/Data Fields')

        da = ds[var_name]

        # the file does not give lat and lon coordinates, only the bounds
        # and resolution, so we need to create them
        f = h5py.File(file, mode='r')
        bounds = eval(f['HDFEOS/GRIDS/ColumnAmountNO2'].attrs['GridSpan'])
        resolution = eval(
            f['HDFEOS/GRIDS/ColumnAmountNO2'].attrs['GridSpacing'])

        lon, lat = define_coordinates(
            lat_bounds=[bounds[2], bounds[3]],
            lon_bounds=[bounds[0], bounds[1]],
            resolution=list(resolution)
        )

        # check that grid variables have the correct shape
        lat_len = f['HDFEOS/GRIDS/ColumnAmountNO2'].attrs['NumberOfLatitudesInGrid'].item()
        lon_len = f['HDFEOS/GRIDS/ColumnAmountNO2'].attrs['NumberOfLongitudesInGrid'].item()

        assert lat_len == len(
            lat), f"lat length {lat_len} does not match grid length {len(lat)}"
        assert lon_len == len(
            lon), f"lon length {lon_len} does not match grid length {len(lon)}"

        # add the coordinates to the data array
        da = da.rename({'phony_dim_0': 'lat', 'phony_dim_1': 'lon'})
        da = da.assign_coords(lat=('lat', lat), lon=('lon', lon))

        # crop data to area of interest (with some buffer)
        da = da.sel(lon=slice(grid_bounds[0]-1, grid_bounds[2]+1),
                    lat=slice(grid_bounds[1]-1, grid_bounds[3]+1))

        # # # get the date string

        year_str = f['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES'].attrs['GranuleYear'].item()
        month_str = f['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES'].attrs['GranuleMonth'].item()
        day_str = f['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES'].attrs['GranuleDay'].item()

        # convert to datetime
        datestamp = pd.to_datetime(f"{year_str}-{month_str}-{day_str}")

        # convert to datestring in format YYYY-MM-DD
        datestr = datestamp.strftime("%Y-%m-%d")

    else:
        raise ValueError(
            f"Unknown variable {var_name} to read from file {file}")

    return da, datestr


def define_coordinates(
    lat_bounds: List[float],
    lon_bounds: List[float],
    resolution: List[float]
):
    """Given latitude bounds, longitude bounds, and the data product 
    resolution, create a meshgrid of points between bounding coordinates.

    This function was copied from https://drivendata.co/blog/predict-no2-benchmark.

    Args:
        lat_bounds (List): latitude bounds as a list.
        lon_bounds (List): longitude bounds as a list.
        resolution (List): data resolution as a list.

    Returns:
        lon (np.array): x (longitude) coordinates.
        lat (np.array): y (latitude) coordinates.
    """
    # Interpolate points between bounds
    # Add 0.125 buffer, source: OMI_L3_ColumnAmountO3.py (HDFEOS script)
    lon = np.arange(lon_bounds[0], lon_bounds[1], resolution[1]) + 0.125
    lat = np.arange(lat_bounds[0], lat_bounds[1], resolution[0]) + 0.125

    return lon, lat


def interp_to_grid(da: xr.DataArray, grid: gpd.GeoDataFrame, var_name: str, datestr: str):

    # interpolate to the grid
    sampled_values = da.interp(
        lon=xr.DataArray(grid["lon"], dims="points"),
        lat=xr.DataArray(grid["lat"], dims="points"),
        method="linear"  # Or "linear" for bilinear interpolation
    )

    grid[var_name] = sampled_values.values

    # add date string
    grid['date'] = datestr

    return grid


if __name__ == "__main__":

    # MERRA AOT
    dataset = 'M2T1NXAER'
    var_name = 'TOTEXTTAU'  # variable name in the source dataset
    var_label = 'aot'  # variable name in the output file
    outfile = 'MERRA2_AOT'
    dataset_version = '5.12.4'  # not needed, but no harm in specifying

    # # MERRA CO
    # dataset = 'M2I3NVCHM'
    # var_name = 'CO'
    # var_label = 'co'
    # outfile = 'MERRA2_CO'
    # dataset_version = '5.12.4' # not needed, but no harm in specifying

    # # OMI NO2
    # dataset = 'OMNO2d'
    # var_name = 'ColumnAmountNO2'
    # var_label = 'omi_no2'
    # outfile = 'OMI_NO2'
    # dataset_version = '003'

    main(dataset=dataset,
         varname=var_name,
         var_label=var_label,
         outfilename=outfile,
         dataset_version=dataset_version,
         start_date="2020-01-01",
         end_date="2020-01-31",)
