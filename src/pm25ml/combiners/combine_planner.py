"""CombineDescriptor for combining data for a specific month."""

from collections.abc import Collection
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from arrow import Arrow

from pm25ml.collectors.collector import UploadResult
from pm25ml.collectors.validate_configuration import VALID_COUNTRIES
from pm25ml.hive_path import HivePath


@dataclass(frozen=True)
class CombinePlan:
    """Descriptor for combining data for a specific month."""

    month: Arrow
    """
    The month in YYYY-MM format that this descriptor applies to.
    """

    paths: AbstractSet[HivePath]
    """
    The paths to the data that should be combined for this month.
    """

    expected_columns: AbstractSet[str]
    """
    The columns that are expected in the combined data for this month.
    This includes both ID columns and value columns.
    """

    @property
    def month_id(self) -> str:
        """The month in 'YYYY-MM' format."""
        return _month_format(self.month)

    @property
    def expected_rows(self) -> int:
        """The number of rows expected in the combined data for this month."""
        return VALID_COUNTRIES["india"] * self.days_in_month

    @property
    def days_in_month(self) -> int:
        """The number of days in the month."""
        return (self.month.ceil("month") - self.month.floor("month")).days + 1


class CombinePlanner:
    """Planner for combining data for multiple months."""

    def __init__(self, months: Collection[Arrow]) -> None:
        """
        Initialize the CombinePlanner.

        :param months: A collection of Arrow objects representing the months to combine.
        """
        self.months = months

    def plan(
        self,
        results: Collection[UploadResult],
    ) -> Collection[CombinePlan]:
        """Choose what to combine for each month."""
        all_id_columns = {
            column for result in results for column in result.pipeline_config.id_columns
        }

        all_value_columns = {
            f"{result.pipeline_config.hive_path.require_key('dataset')}__{column}"
            for result in results
            for column in result.pipeline_config.value_columns
        }

        all_expected_columns = all_id_columns | all_value_columns

        return [
            CombinePlan(
                month=month,
                paths=set(self._list_paths_to_merge(month, results)),
                expected_columns=all_expected_columns,
            )
            for month in self.months
        ]

    def _list_paths_to_merge(
        self,
        month: Arrow,
        results: Collection[UploadResult],
    ) -> Collection[HivePath]:
        hive_paths = [result.pipeline_config.hive_path for result in results]
        year_filter: str = str(month.year)

        month_as_str = _month_format(month)

        month_related = [path for path in hive_paths if path.metadata.get("month") == month_as_str]

        year_related = [path for path in hive_paths if path.metadata.get("year") == year_filter]
        static_related = [path for path in hive_paths if path.metadata.get("type") == "static"]

        return month_related + year_related + static_related


def _month_format(month: Arrow) -> str:
    """
    Format the month in 'YYYY-MM' format.

    :param month: The month to format.
    :return: The formatted month string.
    """
    return month.format("YYYY-MM")
