import numpy as np
from pm25ml.collectors.ned.coord_types import Lat, Lon, NpLat, NpLon


def test__Lat__is_float():
    assert isinstance(Lat(0.0), float)


def test__Lon__is_float():
    assert isinstance(Lon(0.0), float)


def test__NpLat__is_float64():
    assert isinstance(NpLat(np.float64(0.0)), np.float64)


def test__NpLon__is_float64():
    assert isinstance(NpLon(np.float64(0.0)), np.float64)
