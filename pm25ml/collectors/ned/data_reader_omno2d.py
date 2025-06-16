"""Data reader for OMI NO2 data."""

from ast import literal_eval
from typing import TYPE_CHECKING, cast

import arrow
import numpy as np
import xarray
from fsspec.spec import AbstractBufferedFile
from numpy.typing import NDArray

from pm25ml.collectors.ned.coord_types import Lat, Lon, NpLat, NpLon
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
        """
        Extract the data from an OMI NO2 file for a dataset.

        Args:
            file (AbstractBufferedFile): The file containing the OMI NO2 data.
            dataset_descriptor (NedDatasetDescriptor): The dataset descriptor containing metadata
            on how to extract the data.

        Returns:
            NedDayData: An object containing the results of the extraction.

        """
        date = self._extract_date(file)

        lon, lat = self._build_coords(file)

        ds = xarray.open_dataset(
            cast("ReadBuffer", file),
            group="HDFEOS/GRIDS/ColumnAmountNO2/Data Fields",
        )

        var_name = dataset_descriptor.source_variable_name
        min_lon = dataset_descriptor.filter_min_lon
        max_lon = dataset_descriptor.filter_max_lon
        min_lat = dataset_descriptor.filter_min_lat
        max_lat = dataset_descriptor.filter_max_lat

        data_array = ds[var_name]
        data_array = data_array.rename({"phony_dim_0": "lat", "phony_dim_1": "lon"})
        data_array = data_array.assign_coords(lat=("lat", lat), lon=("lon", lon))

        data_array = data_array.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(min_lat, max_lat),
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
        bounds: tuple[Lon, Lon, Lat, Lat] = literal_eval(grid_info.attrs["GridSpan"])
        resolution: tuple[Lon, Lat] = literal_eval(grid_info.attrs["GridSpacing"])
        min_lon, max_lon, min_lat, max_lat = bounds

        lon, lat = self._define_coords(
            lon_bounds=(min_lon, max_lon),
            lat_bounds=(min_lat, max_lat),
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
        *,
        lat_bounds: tuple[Lat, Lat],
        lon_bounds: tuple[Lon, Lon],
        resolution: tuple[Lon, Lat],
    ) -> tuple[NDArray[NpLon], NDArray[NpLat]]:
        """
        Create a meshgrid of points between bounding coordinates.

        Uses latitude bounds, longitude bounds, and the data product
        resolution to create a grid of points.

        Args:
            lat_bounds (tuple[Lat, Lat]): latitude bounds as a tuple.
            lon_bounds (tuple[Lon, Lon]): longitude bounds as a tuple.
            resolution (tuple[Lon, Lat]): data resolution as a tuple.

        Returns:
            lon (NDArray[NpLon]): x (longitude) coordinates.
            lat (NDArray[NpLat]): y (latitude) coordinates.

        """
        # Interpolate points between bounds

        lon_centre_adjustment = resolution[1] / 2
        lat_centre_adjustment = resolution[0] / 2

        lon = cast(
            "NDArray[NpLon]",
            np.arange(lon_bounds[0], lon_bounds[1], resolution[1]) + lon_centre_adjustment,
        )
        lat = cast(
            "NDArray[NpLat]",
            np.arange(lat_bounds[0], lat_bounds[1], resolution[0]) + lat_centre_adjustment,
        )

        return lon, lat
