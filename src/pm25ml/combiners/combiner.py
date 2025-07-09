"""Combines monthly data from the archive storage into a single dataset."""

from collections.abc import Collection

from pm25ml.combiners.archive_wide_combiner import ArchiveWideCombiner
from pm25ml.combiners.combine_planner import CombinePlan
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.logging import logger


class MonthlyValidationError(Exception):
    """An error raised when the combined monthly data does not match expectations."""


class MonthlyCombiner:
    """Combines monthly data from the archive storage into a single dataset."""

    def __init__(
        self,
        combined_storage: CombinedStorage,
        archived_wide_combiner: ArchiveWideCombiner,
    ) -> None:
        """
        Initialize the MonthlyCombiner.

        :param combined_storage: The storage where the combined results will be stored.
        :param archived_wide_combiner: The combiner that handles wide archive data.
        """
        self.combined_storage = combined_storage
        self.archived_wide_combiner = archived_wide_combiner

    def combine_for_months(
        self,
        combine_descriptors: Collection[CombinePlan],
    ) -> None:
        """
        For each month available in the archive storage, combine the data from all processors.

        :param months: A collection of Arrow objects representing the months to combine.
        :param processors: A collection of ExportPipeline instances to process the data for.
        """
        for desc in combine_descriptors:
            month_short = desc.month_id
            if self._needs_combining(
                desc,
            ):
                # This needs to be all processors, not just the filtered ones.
                self.archived_wide_combiner.combine(
                    desc,
                )
                logger.info(
                    f"Combined data for month {month_short} into wide format.",
                )
                self._validate_combined(desc)
            else:
                logger.info(
                    f"Skipping combining for month {month_short} as it is already combined.",
                )

    def _needs_combining(self, desc: CombinePlan) -> bool:
        logger.debug(
            f"Checking if data for month {desc.month_id} needs to be combined.",
        )
        exists_in_storage = self.combined_storage.does_dataset_exist(
            result_subpath=f"stage=combined_monthly/month={desc.month_id}",
        )
        if not exists_in_storage:
            return True

        try:
            self._validate_combined(
                desc,
            )
        except MonthlyValidationError as exc:
            logger.debug(
                f"Data doesn't match expected schema for month {desc.month_id}, "
                f" requesting re-combination: {exc}",
                exc_info=True,
            )
            return True

        return False

    def _validate_combined(self, desc: CombinePlan) -> None:
        expected_rows = desc.expected_rows
        all_expected_columns = set(desc.expected_columns)
        month_short = desc.month_id
        logger.debug(
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
            raise MonthlyValidationError(msg)

        missing = all_expected_columns - set(final_combined.columns)
        if missing:
            msg = (
                f"Expected columns {all_expected_columns} in the final combined result, "
                f"but {missing} were missing."
            )
            raise MonthlyValidationError(msg)
