"""Combine data from multiple sources in the archive into a single file."""

from concurrent.futures import ThreadPoolExecutor
from re import match

from polars import DataFrame

from pm25ml.collectors.archive_storage import IngestArchiveStorage, IngestDataAsset
from pm25ml.collectors.export_pipeline import ExportPipeline
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.logging import logger

INDEX_COLUMNS = ["grid_id", "date"]


class ArchiveWideCombiner:
    """Combines data from multiple sources in the archive into a single file."""

    def __init__(
        self,
        archive_storage: IngestArchiveStorage,
        combined_storage: CombinedStorage,
    ) -> None:
        """
        Initialize the ArchiveWideCombiner with the archive storage.

        :param archive_storage: The IngestArchiveStorage instance to use for storage operations.
        :param combined_storage: The CombinedStorage instance to use for writing combined data.
        """
        self.archive_storage = archive_storage
        self.combined_storage = combined_storage

    def combine(self, month: str, processors: list[ExportPipeline]) -> None:
        """
        Combine data from multiple sources in the archive for a given month.

        We need the processors to determine which data to combine - not just choosing from the
        data available as some of that might be deprecated or unsynced data.

        :param month: The month in 'YYYY-MM' format for which to combine data.
        :param processors: A list of ExportPipeline instances to process the data for.
        :raises ValueError: If the month does not match the expected 'YYYY-MM' format
        """
        # Check month matches YYYY-MM format
        self._check_month_arg(month)

        all_to_merge = self._list_paths_to_merge(month, processors)

        if not all_to_merge:
            msg = (
                f"No data found for month '{month}'. "
                "Ensure that the month is correct and that data exists in the archive."
            )
            raise ValueError(msg)

        logger.info(
            f"Loading data to combine for month {month} from {len(all_to_merge)} paths",
        )
        loaded_assets = [self.archive_storage.read_data_asset(path) for path in all_to_merge]

        with_renamed_value_columns = self._add_dataset_to_value_columns(loaded_assets)

        logger.info(
            f"Normalising index columns for {len(loaded_assets)} tables",
        )
        normalised_tables = self._normalise_index_columns(with_renamed_value_columns)

        logger.info(
            f"Combining {len(normalised_tables)} tables into a single table",
        )
        combined_table = self._parallel_inner_join(normalised_tables)

        logger.info(
            f"Writing combined table with {combined_table.height} rows and "
            f"{len(combined_table.columns)} columns to storage",
        )
        result_subpath = f"stage=combined_monthly/month={month}"
        self.combined_storage.write_to_destination(
            table=combined_table,
            result_subpath=result_subpath,
        )

    def _list_paths_to_merge(self, month: str, processors: list[ExportPipeline]) -> list[str]:
        hive_paths = [processor.get_config_metadata().hive_path for processor in processors]
        year_filter = month[:4]

        month_related = [path for path in hive_paths if path.metadata.get("month") == month]

        year_related = [path for path in hive_paths if path.metadata.get("year") == year_filter]
        static_related = [path for path in hive_paths if path.metadata.get("type") == "static"]

        filtered_paths = month_related + year_related + static_related

        return [path.result_subpath for path in filtered_paths]

    def _check_month_arg(self, month: str) -> None:
        regex = r"^\d{4}-\d{2}$"
        if not match(regex, month):
            msg = f"Month '{month}' does not match the expected format 'YYYY-MM'."
            raise ValueError(msg)

    @staticmethod
    def _normalise_index_columns(tables: list[DataFrame]) -> list[DataFrame]:
        """
        Normalize the index columns of a list of DataFrames.

        :param tables: A list of DataFrames to normalize.
        :return: A list of DataFrames with normalized index columns.
        """

        # The date column might be in the format "YYYY-MM-DD" or "YYYY-MM-DDT00:00:00"
        # we just want the "YYYY-MM-DD" part.
        def _normalize_date_column(table: DataFrame) -> DataFrame:
            """
            Normalize the 'date' column in a DataFrame to ensure it is in 'YYYY-MM-DD' format.

            :param table: The DataFrame to normalize.
            :return: The DataFrame with the 'date' column normalized.
            """
            if "date" in table.columns:
                table = table.with_columns(
                    table["date"].str.slice(0, 10).alias("date"),
                )
            return table

        return [table.pipe(_normalize_date_column) for table in tables]

    @staticmethod
    def _parallel_inner_join(initial_tables: list[DataFrame]) -> DataFrame:
        """
        Perform a parallel inner join on a list of DataFrames.

        This function uses a thread pool to join pairs of DataFrames until only one remains.

        :param tables: A list of DataFrames to join.
        :return: A single DataFrame resulting from the inner joins of all input DataFrames.
        :raises ValueError: If the final result does not contain exactly one DataFrame.
        """

        def _join_two_tables(left_table: DataFrame, right_table: DataFrame) -> DataFrame:
            """
            Join two DataFrames on the 'grid_id' and 'date' columns using an inner join.

            It only joins on the common join keys that are present in both DataFrames.

            :param left_table: The left DataFrame to join.
            :param right_table: The right DataFrame to join.
            :return: A DataFrame resulting from the inner join of the two input DataFrames.
            """
            left_join_keys = set(left_table.columns).intersection(INDEX_COLUMNS)
            right_join_keys = set(right_table.columns).intersection(INDEX_COLUMNS)
            matching_keys = set(left_join_keys).intersection(right_join_keys)

            return left_table.join(
                right_table,
                on=list(matching_keys),
                how="inner",
            )

        remaining = initial_tables
        with ThreadPoolExecutor() as executor:
            # Initial join operations
            while len(remaining) > 1:
                # Pair tables for joining
                table_pairs = list(zip(remaining[0::2], remaining[1::2]))
                leftover = [remaining[-1]] if len(remaining) % 2 == 1 else []
                combined_pairs = list(
                    executor.map(
                        lambda pair: _join_two_tables(*pair),
                        table_pairs,
                    ),
                )
                remaining = combined_pairs + leftover

        return remaining[0]

    def _add_dataset_to_value_columns(self, tables: list[IngestDataAsset]) -> list[DataFrame]:
        """
        Rename columns in each DataFrame to include the dataset name extracted from the asset path.

        :param tables: A list of DataFrames to process.
        :return: A list of DataFrames with renamed columns.
        """

        def _rename_columns_with_dataset(table: DataFrame, dataset_name: str) -> DataFrame:
            """
            Rename columns in a DataFrame to include the dataset name.

            :param table: The DataFrame to rename.
            :param dataset_name: The dataset name to prepend to column names.
            :return: The DataFrame with renamed columns.
            """
            renamed_columns = {
                col: f"{dataset_name}__{col}" for col in table.columns if col not in INDEX_COLUMNS
            }
            return table.rename(renamed_columns)

        renamed_tables = []
        for asset in tables:
            dataset_name = asset.hive_path.require_key("dataset")
            # Extract dataset name using HivePath's metadata
            renamed_tables.append(_rename_columns_with_dataset(asset.data_frame, dataset_name))

        return renamed_tables
