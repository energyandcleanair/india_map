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

    """Extracts the data from a MERRA-2 file for a dataset."""

    def extract_data(
        self,
        file: AbstractBufferedFile,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> NedDayData:
        """Fetch data for the given date range."""
        ds = xarray.open_dataset(cast("ReadBuffer", file), chunks="auto", engine="h5netcdf")

        begin_date = ds.attrs.get("RangeBeginningDate")
        if not begin_date:
            msg = "Dataset does not contain a valid 'RangeBeginningDate' attribute."
            raise ValueError(msg)

        var_name = dataset_descriptor.source_variable_name
        filter_bounds = dataset_descriptor.filter_bounds

        ds = ds[var_name].sel(
            lon=slice(filter_bounds[0], filter_bounds[2]),
            lat=slice(filter_bounds[1], filter_bounds[3]),
        )

        if "lev" in ds.dims:
            ds = ds.isel(lev=-1) if ds.lev.attrs["positive"] == "down" else ds.isel(lev=0)

        ds = ds.mean(dim="time", keep_attrs=True)

        expected_dimensions = 2
        if len(ds.dims) != expected_dimensions:
            msg = f"Data is not 2D for projection: dimensions are {ds.dims}"
            raise ValueError(msg)
        return NedDayData(data_array=ds, date=begin_date)
