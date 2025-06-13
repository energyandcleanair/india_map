"""Defines data reader contracts for reading data from NASA EarthData."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray
    from fsspec.spec import AbstractBufferedFile

    from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor


class NedDayData:
    """
    Data for a single day.

    This class encapsulates the data for a single day, including the data array
    and the date in 'YYYY-MM-DD' format.
    """

    def __init__(self, data_array: xarray.DataArray, date: str) -> None:
        """
        Initialize the NedDayData instance.

        Args:
            data_array (xarray.DataArray): The data array containing the data for the day.
            date (str): The date in 'YYYY-MM-DD' format.

        """
        self.data = data_array
        self.date = date


class NedDataReader:
    """Extracts the data from a file for a dataset."""

    def __init__(self) -> None:
        """Initialize the data provider."""

    def extract_data(
        self,
        file: AbstractBufferedFile,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> NedDayData:
        """Fetch data for the given date range."""
        msg = "This method should be implemented by subclasses."
        raise NotImplementedError(msg)
