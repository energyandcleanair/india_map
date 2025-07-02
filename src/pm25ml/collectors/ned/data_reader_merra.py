"""Data reader for MERRA-2 data files."""

from typing import IO, ClassVar

import xarray

from pm25ml.collectors.ned.data_readers import NedDataReader, NedDayData
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor


class MerraDataReader(NedDataReader):
    """Data reader for MERRA-2 data files."""

    expected_dimensions: ClassVar[list[str]] = [
        "lon",
        "lat",
        "time",
    ]
    optional_dimensions: ClassVar[list[str]] = ["lev"]
    all_dimensions: ClassVar[list[str]] = expected_dimensions + optional_dimensions

    def __init__(self) -> None:
        """Initialize the data reader."""
        super().__init__()

    def extract_data(
        self,
        file: IO[bytes],
        dataset_descriptor: NedDatasetDescriptor,
    ) -> NedDayData:
        """
        Extract the data from a MERRA-2 file for a dataset.

        Args:
            file (IO[bytes]): The file containing the MERRA-2 data.
            dataset_descriptor (NedDatasetDescriptor): The dataset descriptor containing metadata
            on how to extract the data.

        Returns:
            NedDayData: An object containing the results of the extraction.

        """
        dataset = xarray.open_dataset(file, chunks="auto", engine="h5netcdf")

        self._check_expected_dimensions(dataset)

        begin_date = dataset.attrs.get("RangeBeginningDate")
        if not begin_date:
            msg = "Dataset does not contain a valid 'RangeBeginningDate' attribute."
            raise ValueError(msg)

        # Only supports one variable for retrieval and extracting
        if len(dataset_descriptor.variable_mapping) != 1:
            msg = (
                "MERRA-2 data reader only supports one variable for retrieval. "
                f"Provided variables: {dataset_descriptor.variable_mapping.keys()}"
            )
            raise ValueError(msg)

        var_name = next(iter(dataset_descriptor.variable_mapping.keys()))
        min_lon = dataset_descriptor.filter_min_lon
        max_lon = dataset_descriptor.filter_max_lon
        min_lat = dataset_descriptor.filter_min_lat
        max_lat = dataset_descriptor.filter_max_lat

        dataset = dataset[[var_name]].sel(
            lon=slice(min_lon, max_lon),
            lat=slice(min_lat, max_lat),
        )

        # let's do some checking that if lev i
        lev_selector = dataset_descriptor.level

        if lev_selector is None and "lev" in dataset.dims:
            msg = (
                "Dataset contains 'lev' dimension, but no level specified in the descriptor. "
                "Please specify a level or remove the 'lev' dimension from the dataset."
            )
            raise ValueError(msg)

        if lev_selector is not None and "lev" not in dataset.dims:
            msg = (
                "Dataset does not contain 'lev' dimension, but a level is specified in "
                "the descriptor. Please remove the level specification or ensure the dataset "
                "contains 'lev' dimension."
            )
            raise ValueError(msg)

        if "lev" in dataset.dims:
            dataset = dataset.isel(lev=lev_selector)

        dataset = dataset.mean(dim="time", keep_attrs=True)

        expected_dimensions = 2
        if len(dataset.dims) != expected_dimensions:
            msg = f"Data is not 2D for projection: dimensions are {dataset.dims}"
            raise ValueError(msg)
        return NedDayData(dataset=dataset, date=begin_date)

    def _check_expected_dimensions(
        self,
        dataset: xarray.Dataset,
    ) -> None:
        # Check if all expected dimensions are present
        actual_dimensions = list(dataset.sizes.keys())
        missing_expected_dimensions = [
            dim for dim in self.expected_dimensions if dim not in actual_dimensions
        ]
        if missing_expected_dimensions:
            msg = (
                f"Dataset is missing expected dimensions: {missing_expected_dimensions}. "
                f"Actual dimensions are: {actual_dimensions}. "
                f"Required dimensions are: {self.expected_dimensions}."
            )
            raise ValueError(msg)

        # Check if any unexpected dimensions are present
        unexpected_dimensions = [dim for dim in actual_dimensions if dim not in self.all_dimensions]
        if unexpected_dimensions:
            msg = (
                f"Dataset contains unexpected dimensions: {unexpected_dimensions}. "
                f"Actual dimensions are: {actual_dimensions}. "
                f"Allowable dimensions are: {self.all_dimensions}."
            )
            raise ValueError(msg)
