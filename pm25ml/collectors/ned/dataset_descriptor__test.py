import pytest
from arrow import Arrow
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.coord_types import Lat, Lon


@pytest.fixture
def descriptor():
    return NedDatasetDescriptor(
        dataset_name="NED",
        dataset_version="v1.0",
        start_date=Arrow(2023, 1, 1),
        end_date=Arrow(2023, 1, 10),
        filter_bounds=(Lon(70.0), Lat(8.0), Lon(90.0), Lat(37.0)),
        variable_mapping={
            "elevation": "elev",
        },
        level=1,
    )


def test__init__valid_args__attributes_set_correctly(descriptor):
    assert descriptor.dataset_name == "NED"
    assert descriptor.dataset_version == "v1.0"
    assert descriptor.start_date == Arrow(2023, 1, 1)
    assert descriptor.end_date == Arrow(2023, 1, 10)
    assert descriptor.filter_bounds == (Lon(70.0), Lat(8.0), Lon(90.0), Lat(37.0))
    assert descriptor.variable_mapping["elevation"] == "elev"
    assert descriptor.level == 1


def test__days_in_range__returns_correct_number_of_days(descriptor):
    assert descriptor.days_in_range == 10


def test__filter_min_lon__returns_first_element(descriptor):
    assert descriptor.filter_min_lon == Lon(70.0)


def test__filter_min_lat__returns_second_element(descriptor):
    assert descriptor.filter_min_lat == Lat(8.0)


def test__filter_max_lon__returns_third_element(descriptor):
    assert descriptor.filter_max_lon == Lon(90.0)


def test__filter_max_lat__returns_fourth_element(descriptor):
    assert descriptor.filter_max_lat == Lat(37.0)
