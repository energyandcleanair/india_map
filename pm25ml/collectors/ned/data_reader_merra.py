"""Data reader for MERRA-2 data files."""

from typing import TYPE_CHECKING, ClassVar, cast

import xarray
from fsspec.spec import AbstractBufferedFile

from pm25ml.collectors.ned.data_readers import NedDataReader, NedDayData
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor

if TYPE_CHECKING:
    from xarray.core.types import ReadBuffer


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
        file: AbstractBufferedFile,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> NedDayData:
        """
        Extract the data from a MERRA-2 file for a dataset.

        Args:
            file (AbstractBufferedFile): The file containing the MERRA-2 data.
            dataset_descriptor (NedDatasetDescriptor): The dataset descriptor containing metadata
            on how to extract the data.

        Returns:
            NedDayData: An object containing the results of the extraction.

        """
        dataset = xarray.open_dataset(cast("ReadBuffer", file), chunks="auto", engine="h5netcdf")

        self._check_expected_dimensions(dataset)

        begin_date = dataset.attrs.get("RangeBeginningDate")
        if not begin_date:
            msg = "Dataset does not contain a valid 'RangeBeginningDate' attribute."
            raise ValueError(msg)

        var_name = dataset_descriptor.source_variable_name
        min_lon = dataset_descriptor.filter_min_lon
        max_lon = dataset_descriptor.filter_max_lon
        min_lat = dataset_descriptor.filter_min_lat
        max_lat = dataset_descriptor.filter_max_lat

        data_array = dataset[var_name].sel(
            lon=slice(min_lon, max_lon),
            lat=slice(min_lat, max_lat),
        )

        if "lev" in data_array.dims:
            data_array = (
                data_array.isel(lev=-1)
                if data_array.lev.attrs["positive"] == "down"
                else data_array.isel(lev=0)
            )

        data_array = data_array.mean(dim="time", keep_attrs=True)

        expected_dimensions = 2
        if len(data_array.dims) != expected_dimensions:
            msg = f"Data is not 2D for projection: dimensions are {data_array.dims}"
            raise ValueError(msg)
        return NedDayData(data_array=data_array, date=begin_date)

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
