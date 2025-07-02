"""Defines data reader contracts for reading data from NASA EarthData."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    import xarray

    from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor


class NedDayData:
    """
    Data for a single day.

    This class encapsulates the data for a single day, including the data array
    and the date in 'YYYY-MM-DD' format.
    """

    def __init__(self, dataset: xarray.Dataset, date: str) -> None:
        """
        Initialize the NedDayData instance.

        Args:
            dataset: The dataset containing the data for the day.
            date: The date in 'YYYY-MM-DD' format.

        """
        self.data = dataset
        self.date = date


class NedDataReader:
    """Extracts the data from a file for a dataset."""

    def __init__(self) -> None:
        """Initialize the data provider."""

    def extract_data(
        self,
        file: IO[bytes],
        dataset_descriptor: NedDatasetDescriptor,
    ) -> NedDayData:
        """
        Fetch data from a file for a dataset.

        It must not rename the variables in the dataset to the target variable names.

        Args:
            file (IO[bytes]): The file containing the data.
            dataset_descriptor (NedDatasetDescriptor): The dataset descriptor containing metadata
            and processing instructions.

        Returns:
            NedDayData: An object containing the results of the extraction.

        """
        msg = "This method should be implemented by subclasses."
        raise NotImplementedError(msg)
