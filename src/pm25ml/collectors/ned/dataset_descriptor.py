"""NED dataset descriptor module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from arrow import Arrow

    from pm25ml.collectors.ned.coord_types import Lat, Lon

InterpolationMethods = Literal["linear", "nearest"]


@dataclass
class NedDatasetDescriptor:
    """
    Descriptor for the NED dataset.

    This class provides metadata needed to identify, subset, reduce, and regrid the dataset.

    It only supports a single variable to extract from the dataset.
    """

    dataset_name: str
    dataset_version: str
    start_date: Arrow
    end_date: Arrow
    filter_bounds: tuple[Lon, Lat, Lon, Lat]
    variable_mapping: dict[str, str]
    interpolation_method: InterpolationMethods = "linear"
    level: int | None = None

    def __repr__(self) -> str:
        """Return a string representation of the dataset descriptor."""
        return (
            f"NedDatasetDescriptor(dataset_name={self.dataset_name}, "
            f"dataset_version={self.dataset_version}, "
            f"start_date={self.start_date}, "
            f"end_date={self.end_date}, "
            f"filter_bounds={self.filter_bounds}, "
            f"variable_mapping={self.variable_mapping}, "
            f"level={self.level}, "
            f"interpolation_method={self.interpolation_method})"
        )

    @property
    def days_in_range(self) -> int:
        """Calculate the number of days in the date range."""
        return (self.end_date - self.start_date).days + 1

    @property
    def filter_min_lon(self) -> Lon:
        """The minimum longitude from the filter bounds."""
        return self.filter_bounds[0]

    @property
    def filter_min_lat(self) -> Lat:
        """The minimum latitude from the filter bounds."""
        return self.filter_bounds[1]

    @property
    def filter_max_lon(self) -> Lon:
        """The maximum longitude from the filter bounds."""
        return self.filter_bounds[2]

    @property
    def filter_max_lat(self) -> Lat:
        """The maximum latitude from the filter bounds."""
        return self.filter_bounds[3]
