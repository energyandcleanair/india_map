"""ExportPipeline interface for exporting data."""

from abc import ABC, abstractmethod
from typing import Literal

from pm25ml.hive_path import HivePath

AvailableIdKeys = Literal["date", "grid_id"]
AVAILABLE_ID_KEY_NAMES: list[AvailableIdKeys] = ["date", "grid_id"]


class PipelineConfig:
    """
    Represents the pipeline config metadata of an export operation.

    This is standardised across all export pipelines and can be used to check the
    configuration of the export operation and validate the results after the export
    is complete.
    """

    def __init__(
        self,
        result_subpath: str,
        id_columns: set[AvailableIdKeys],
        value_columns: set[str],
        expected_n_rows: int,
    ) -> None:
        """
        Initialize the ExportResult with the result subpath and expected columns.

        :param result_subpath: The subpath where the result will be stored.
        :param id_columns: The expected ID columns in the result.
        :param value_columns: The expected value columns in the result.
        :param expected_n_rows: The expected number of rows in the result.
        """
        self.result_subpath = result_subpath
        self.id_columns = id_columns
        self.value_columns = value_columns
        self.expected_rows = expected_n_rows

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
