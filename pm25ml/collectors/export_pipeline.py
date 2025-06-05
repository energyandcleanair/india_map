"""ExportPipeline interface for exporting data."""

from abc import ABC, abstractmethod


class ExportPipeline(ABC):
    """
    Abstract base class for export pipelines.

    Responsible for orchestrating the exporting the data from the origin, transforming it into the
    grid format, and uploading it to the underlying storage.
    """

    @abstractmethod
    def upload(self) -> None:
        """Export the given data."""
