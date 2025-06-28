"""NED dataset descriptor module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arrow import Arrow

    from pm25ml.collectors.ned.coord_types import Lat, Lon


class NedDatasetDescriptor:
    """
    Descriptor for the NED dataset.

    This class provides metadata needed to identify, subset, reduce, and regrid the dataset.

    It only supports a single variable to extract from the dataset.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dataset_name: str,
        dataset_version: str,
        start_date: Arrow,
        end_date: Arrow,
        filter_bounds: tuple[Lon, Lat, Lon, Lat],
        variable_mapping: dict[str, str],
        level: int | None = None,
    ) -> None:
        """
        Initialize the dataset descriptor.

        :param dataset_name: Name of the dataset.
        :param dataset_version: Version of the dataset.
        :param start_date: Start date of the dataset to filter the dataset to.
        :param end_date: End date of the dataset to filter the dataset to.
        :param filter_bounds: Geographic bounds for filtering the dataset (west, south, east,
        north).
        :param source_variable_name: Name of the variable in the dataset to be used as source.
        :param target_variable_name: Name of the variable in the dataset to be used as target.
        """
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.start_date = start_date
        self.end_date = end_date
        self.filter_bounds = filter_bounds
        self.variable_mapping = variable_mapping
        self.level = level

    def __repr__(self) -> str:
        """Return a string representation of the dataset descriptor."""
        return (
            f"NedDatasetDescriptor(dataset_name={self.dataset_name}, "
            f"dataset_version={self.dataset_version}, "
            f"start_date={self.start_date}, "
            f"end_date={self.end_date}, "
            f"filter_bounds={self.filter_bounds}, "
            f"variable_mapping={self.variable_mapping}, "
            f"level={self.level})"
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
