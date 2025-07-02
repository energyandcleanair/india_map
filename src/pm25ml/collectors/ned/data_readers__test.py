from unittest.mock import MagicMock
import pytest
import xarray
from pm25ml.collectors.ned.data_readers import NedDayData


def test__NedDayData__data_array_set__data_array_exists():
    mocked_xarray_dataset = MagicMock(spec=xarray.Dataset)
    day_data = NedDayData(dataset=mocked_xarray_dataset, date="2023-01-01")

    assert day_data.data is mocked_xarray_dataset, "Data array should be set correctly."


def test__NedDayData__date_set__date_exists():
    date = "2023-01-01"
    mocked_xarray_dataset = MagicMock(spec=xarray.Dataset)
    day_data = NedDayData(dataset=mocked_xarray_dataset, date=date)

    assert day_data.date == date, "Date should be set correctly."


def test__NedDataReader__extract_data__not_implemented():
    """Test that the extract_data method raises NotImplementedError."""
    from pm25ml.collectors.ned.data_readers import NedDataReader

    reader = NedDataReader()
    with pytest.raises(
        NotImplementedError, match="This method should be implemented by subclasses."
    ):
        reader.extract_data(MagicMock(), MagicMock())
