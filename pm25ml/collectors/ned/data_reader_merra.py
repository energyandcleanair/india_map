"""Data reader for MERRA-2 data files."""

from typing import TYPE_CHECKING, cast

import xarray
from fsspec.spec import AbstractBufferedFile

from pm25ml.collectors.ned.data_readers import NedDataReader, NedDayData
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor

if TYPE_CHECKING:
    from xarray.core.types import ReadBuffer


class MerraDataReader(NedDataReader):
    """Data reader for MERRA-2 data files."""

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

        begin_date = dataset.attrs.get("RangeBeginningDate")
        if not begin_date:
            msg = "Dataset does not contain a valid 'RangeBeginningDate' attribute."
            raise ValueError(msg)

        var_name = dataset_descriptor.source_variable_name
        filter_bounds = dataset_descriptor.filter_bounds

        data_array = dataset[var_name].sel(
            lon=slice(filter_bounds[0], filter_bounds[2]),
            lat=slice(filter_bounds[1], filter_bounds[3]),
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
