"""Validates that the expected results from a dataset match the path's expected results."""

import arrow

from pm25ml.collectors.export_pipeline import ExportPipeline

VALID_COUNTRIES = {
    "india": 33074,
}


def validate_configuration(processors: list[ExportPipeline]) -> None:
    """
    Perform a preliminary check on the expected result.

    checks that the expected result is correct based off the metadata in the path.

    :param n_grids: The number of grids expected in the result.
    :raises ValueError: If the expected number of rows does not match the number of grids.
    """
    for processor in processors:
        _validate_single_processor_config(processor)


def _validate_single_processor_config(processor: ExportPipeline) -> None:
    expected_result = processor.get_config_metadata()

    path_metadata = _PathWithMetadata(
        result_subpath=expected_result.result_subpath,
    )

    # Check that dataset is not empty
    path_metadata.require_key("dataset")
    path_metadata.require_key("country")

    # Check that country is in valid regions
    country = path_metadata.metadata["country"]
    _assert(
        country in VALID_COUNTRIES,
        (
            f"Invalid country '{country}' in {expected_result.result_subpath}. "
            f"Valid countries are: {', '.join(VALID_COUNTRIES.keys())}."
        ),
    )

    n_grids = VALID_COUNTRIES[country]
    expected_rows = _expected_row_count(
        path_metadata.metadata,
        n_grids=n_grids,
    )
    _assert(
        expected_result.expected_rows == expected_rows,
        (
            f"Expected {expected_rows} rows in {expected_result.result_subpath}, "
            f"but found {expected_result.expected_rows} rows based on the metadata."
        ),
    )

    expected_ids = _expected_ids(
        path_metadata.metadata,
    )
    _assert(
        expected_result.id_columns == expected_ids,
        (
            f"Expected ID columns {expected_ids} in {expected_result.result_subpath}, "
            f"but found {expected_result.id_columns}."
        ),
    )


def _expected_row_count(
    path_metadata: dict[str, str],
    n_grids: int,
) -> int:
    if "month" not in path_metadata:
        return n_grids

    month_value = arrow.get(path_metadata["month"])
    end_of_month = month_value.ceil("month")
    days_in_month = (end_of_month - month_value).days + 1

    return days_in_month * n_grids


def _expected_ids(path_metadata: dict[str, str]) -> set[str]:
    if "month" in path_metadata:
        return {"date", "grid_id"}
    return {"grid_id"}


# We use this assert because asserts are stripped out in production code,
# and we want to ensure that the validation logic is always executed.
def _assert(condition: bool, message: str) -> None:  # noqa: FBT001
    """
    Assert that a condition is true, otherwise raise a ValueError with the given message.

    :param condition: The condition to check.
    :param message: The message to include in the ValueError if the condition is false.
    :raises ValueError: If the condition is false.
    """
    if not condition:
        raise ValueError(message)


class _PathWithMetadata:
    """
    A class to represent a path with metadata.

    This class is used to extract metadata from the result subpath.
    """

    def __init__(self, result_subpath: str) -> None:
        self.result_subpath = result_subpath
        self.metadata = self._extract_metadata_from_path(result_subpath)

    @staticmethod
    def _extract_metadata_from_path(
        result_subpath: str,
    ) -> dict[str, str]:
        """
        Extract the details from the result subpath.

        :param result_subpath: The subpath where the result is stored.
        :return: A dictionary containing the keys and values stored in the subpath.
        """
        return dict(part.split("=", 1) for part in result_subpath.split("/") if "=" in part)

    def require_key(self, key: str) -> str:
        """
        Require a key to be present in the metadata.

        :param key: The key to require.
        :return: The value of the key.
        :raises ValueError: If the key is not present in the metadata.
        """
        if key not in self.metadata:
            msg = f"Expected '{key}' key in {self.result_subpath}, but it is missing."
            raise ValueError(msg)
        return self.metadata[key]
