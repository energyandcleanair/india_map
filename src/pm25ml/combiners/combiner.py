"""Combines monthly data from the archive storage into a single dataset."""

from collections.abc import Collection

from arrow import Arrow

from pm25ml.collectors.export_pipeline import ExportPipeline
from pm25ml.collectors.validate_configuration import VALID_COUNTRIES
from pm25ml.combiners.archive_wide_combiner import ArchiveWideCombiner
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.logging import logger


class MonthlyCombiner:
    """Combines monthly data from the archive storage into a single dataset."""

    def __init__(
        self,
        combined_storage: CombinedStorage,
        archived_wide_combiner: ArchiveWideCombiner,
        months: Collection[Arrow],
    ) -> None:
        """
        Initialize the MonthlyCombiner.

        :param combined_storage: The storage where the combined results will be stored.
        :param archived_wide_combiner: The combiner that handles wide archive data.
        """
        self.combined_storage = combined_storage
        self.archived_wide_combiner = archived_wide_combiner
        self.months = months

    def combine_for_months(
        self,
        processors: Collection[ExportPipeline],
    ) -> None:
        """
        For each month available in the archive storage, combine the data from all processors.

        :param months: A collection of Arrow objects representing the months to combine.
        :param processors: A collection of ExportPipeline instances to process the data for.
        """
        for month in self.months:
            if self.archived_wide_combiner.needs_combining(
                month=self._to_month_short(month),
            ):
                # This needs to be all processors, not just the filtered ones.
                self.archived_wide_combiner.combine(
                    month=self._to_month_short(month),
                    processors=processors,
                )

            self._validate_combined(processors, month)

    def _validate_combined(self, processors: Collection[ExportPipeline], month: Arrow) -> None:
        month_short = self._to_month_short(month)
        dates_in_month: list[Arrow] = list(
            Arrow.range("day", start=month, end=month.shift(months=1).shift(days=-1)),
        )

        all_id_columns = {
            column
            for processor in processors
            for column in processor.get_config_metadata().id_columns
        }

        all_value_columns = {
            f"{processor.get_config_metadata().hive_path.require_key('dataset')}__{column}"
            for processor in processors
            for column in processor.get_config_metadata().value_columns
        }

        all_expected_columns = all_id_columns | all_value_columns

        expected_rows = VALID_COUNTRIES["india"] * len(dates_in_month)
        logger.info(
            f"Validating final combined result with {expected_rows} "
            f"expected rows and {len(all_expected_columns)} expected columns",
        )
        final_combined = self.combined_storage.read_dataframe(
            result_subpath=f"stage=combined_monthly/month={month_short}",
        )

        # Validate final combined result has expected rows and columns
        if final_combined.shape[0] != expected_rows:
            msg = (
                f"Expected {expected_rows} rows in the final combined result, "
                f"but found {final_combined.shape[0]} rows."
            )
            raise ValueError(msg)

        missing = all_expected_columns - set(final_combined.columns)
        if missing:
            msg = (
                f"Expected columns {all_expected_columns} in the final combined result, "
                f"but {missing} were missing."
            )
            raise ValueError(msg)

    @staticmethod
    def _to_month_short(month: Arrow) -> str:
        return month.format("YYYY-MM")
