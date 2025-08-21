"""Defines the protocol for results writing."""

from typing import Protocol

from pm25ml.collectors.geo_time_grid_dataset import GeoTimeGridDataset


class FinalResultWriter(Protocol):
    """A protocol for writing final results."""

    def write(self, result: GeoTimeGridDataset) -> None:
        """Write the final model output to the storage."""
