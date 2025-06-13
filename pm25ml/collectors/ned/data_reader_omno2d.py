"""Data reader for OMI NO2 data."""

from ast import literal_eval
from typing import TYPE_CHECKING, cast

import arrow
import numpy as np
import xarray
from fsspec.spec import AbstractBufferedFile

from pm25ml.collectors.ned.data_readers import NedDataReader, NedDayData
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor

if TYPE_CHECKING:
    from xarray.core.types import ReadBuffer


class Omno2dReader(NedDataReader):
    """
    Data reader for OMI NO2 data.

    This class reads OMI NO2 data from an HDF5 file and extracts the relevant data
    for a given date range and geographical bounds.
    """

    def __init__(self) -> None:
        """Initialize the OMI data reader."""
        super().__init__()

    def extract_data(
        self,
        file: AbstractBufferedFile,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> NedDayData:
        """Read the OMI NO2 data from the file."""
        date = self._extract_date(file)

        lon, lat = self._build_coords(file)

        ds = xarray.open_dataset(
            cast("ReadBuffer", file),
            group="HDFEOS/GRIDS/ColumnAmountNO2/Data Fields",
        )

        var_name = dataset_descriptor.source_variable_name
        filter_bounds = dataset_descriptor.filter_bounds

        data_array = ds[var_name]
        data_array = data_array.rename({"phony_dim_0": "lat", "phony_dim_1": "lon"})
        data_array = data_array.assign_coords(lat=("lat", lat), lon=("lon", lon))

        data_array = data_array.sel(
            lon=slice(filter_bounds[0], filter_bounds[2]),
            lat=slice(filter_bounds[1], filter_bounds[3]),
        )

        return NedDayData(data_array=data_array, date=date)

    def _extract_date(self, file: AbstractBufferedFile) -> str:
        file_attributes = xarray.open_dataset(
            cast("ReadBuffer", file),
            group="HDFEOS/ADDITIONAL/FILE_ATTRIBUTES",
        )

        year_str = file_attributes.attrs["GranuleYear"].item()
        month_str = file_attributes.attrs["GranuleMonth"].item()
        day_str = file_attributes.attrs["GranuleDay"].item()

        return arrow.get(int(year_str), int(month_str), int(day_str)).format("YYYY-MM-DD")

    def _build_coords(self, file: AbstractBufferedFile) -> tuple[np.ndarray, np.ndarray]:
        grid_info = xarray.open_dataset(
            cast("ReadBuffer", file),
            group="HDFEOS/GRIDS/ColumnAmountNO2",
        )
        bounds: tuple[float, float, float, float] = literal_eval(grid_info.attrs["GridSpan"])
        resolution: tuple[float, float] = literal_eval(grid_info.attrs["GridSpacing"])

        lon, lat = self._define_coords(
            lat_bounds=(bounds[2], bounds[3]),
            lon_bounds=(bounds[0], bounds[1]),
            resolution=resolution,
        )

        lat_len = grid_info.attrs["NumberOfLatitudesInGrid"].item()
        lon_len = grid_info.attrs["NumberOfLongitudesInGrid"].item()

        if lat_len != len(lat):
            msg = f"lat length {lat_len} does not match generated grid length {len(lat)}"
            raise ValueError(msg)
        if lon_len != len(lon):
            msg = f"lon length {lon_len} does not match generated grid length {len(lon)}"
            raise ValueError(msg)

        return lon, lat

    def _define_coords(
        self,
        lat_bounds: tuple[float, float],
        lon_bounds: tuple[float, float],
        resolution: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a meshgrid of points between bounding coordinates.

        Uses latitude bounds, longitude bounds, and the data product
        resolution to create a grid of points.

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
