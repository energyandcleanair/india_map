"""ExportPipeline interface for exporting data."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pm25ml.hive_path import HivePath

AvailableIdKeys = Literal["date", "grid_id"]
AVAILABLE_ID_KEY_NAMES: list[AvailableIdKeys] = ["date", "grid_id"]


class MissingDataError(Exception):
    """
    Exception raised when the export pipeline is missing data.

    This exception is raised when the export pipeline cannot find the expected data
    for the export operation.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the MissingDataError with a message.

        :param message: The error message to be displayed.
        """
        super().__init__(message)


class ErrorWhileFetchingDataError(Exception):
    """
    Exception raised when there is an error while fetching data for the export pipeline.

    This exception is raised when the export pipeline encounters an error while trying to
    fetch the data required for the export operation.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the ErrorWhileFetchingDataError with a message.

        :param message: The error message to be displayed.
        """
        super().__init__(message)


class MissingDataHeuristic(Enum):
    """
    Enum representing the heuristic used to determine if data is missing.

    This is used to define how the export pipeline should handle missing data.
    """

    FAIL = ("fail", False)
    """
    Fail the export operation if data is missing.
    """

    COPY_LATEST_AVAILABLE = ("copy_latest_available", True)
    """
    Copy the latest available data if missing.
    """

    def __init__(self, type_name: str, allows_missing: bool) -> None:  # noqa: FBT001
        """
        Initialize the MissingDataHeuristic with a name and allows_missing flag.

        :param type_name: The name of the heuristic.
        :param allows_missing: Whether the heuristic allows missing data.
        """
        self.type_name = type_name
        self.allows_missing = allows_missing


@dataclass(frozen=True)
class PipelineConfig:
    """
    Represents the pipeline config metadata of an export operation.

    This is standardised across all export pipelines and can be used to check the
    configuration of the export operation and validate the results after the export
    is complete.
    """

    result_subpath: str
    """
    The subpath where the result will be stored.
    """
    id_columns: set[AvailableIdKeys]
    """
    The expected ID columns in the result.
    """
    value_columns: set[str]
    """
    The expected value columns in the result.
    """
    expected_rows: int
    """
    The expected number of rows in the result.
    """
    missing_data_heuristic: MissingDataHeuristic = MissingDataHeuristic.FAIL

    @property
    def all_columns(self) -> set[str]:
        """
        Get all columns in the result.

        :return: A set of all columns in the result.
        """
        return self.id_columns.union(self.value_columns)

    @property
    def hive_path(self) -> HivePath:
        """
        Get the HivePath representation of the result subpath.

        :return: A HivePath object representing the result subpath.
        """
        return HivePath(self.result_subpath)


class ExportPipeline(ABC):
    """
    Abstract base class for export pipelines.

    Responsible for orchestrating the exporting the data from the origin, transforming it into the
    grid format, and uploading it to the underlying storage.
    """

    @abstractmethod
    def upload(self) -> None:
        """Export the given data."""

    @abstractmethod
    def get_config_metadata(self) -> PipelineConfig:
        """
        Get the expected result of the export operation.

        This method should be overridden by subclasses to provide the expected result.
        """
