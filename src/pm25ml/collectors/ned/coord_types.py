"""Coordinate types for NED."""

from typing import NewType

import numpy as np

Lon = NewType("Lon", float)
Lat = NewType("Lat", float)
NpLon = NewType("NpLon", np.float64)
NpLat = NewType("NpLat", np.float64)
