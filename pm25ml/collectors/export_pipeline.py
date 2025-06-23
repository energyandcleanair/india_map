"""ExportPipeline interface for exporting data."""

from abc import ABC, abstractmethod
from typing import Literal

AvailableIdKeys = Literal["date", "grid_id"]
AVAILABLE_ID_KEY_NAMES: list[AvailableIdKeys] = ["date", "grid_id"]


class ExportResult:
    """Represents the result of an export operation."""

    def __init__(
        self,
        result_subpath: str,
        expected_id_columns: set[AvailableIdKeys],
        expected_value_columns: set[str],
    ) -> None:
        """
        Initialize the ExportResult with the result subpath and expected columns.

        :param result_subpath: The subpath where the result will be stored.
        :param expected_id_columns: The expected ID columns in the result.
        :param expected_value_columns: The expected value columns in the result.
        """
        self.result_subpath = result_subpath
        self.expected_id_columns = expected_id_columns
        self.expected_value_columns = expected_value_columns


class ExportPipeline(ABC):
    """
    Abstract base class for export pipelines.

    Responsible for orchestrating the exporting the data from the origin, transforming it into the
    grid format, and uploading it to the underlying storage.
    """

    @abstractmethod
    def upload(self) -> ExportResult:
        """Export the given data."""
