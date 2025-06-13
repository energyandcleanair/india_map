"""NED dataset descriptor module."""

from arrow import Arrow


class NedDatasetDescriptor:
    """
    Descriptor for the NED dataset.

    This class provides metadata needed to identify, subset, reduce, and regrid the dataset.

    It only supports a single variable to extract from the dataset.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dataset_name: str,
        dataset_version: str,
        start_date: Arrow,
        end_date: Arrow,
        filter_bounds: tuple[float, float, float, float],
        source_variable_name: str,
        target_variable_name: str,
    ) -> None:
        """
        Initialize the dataset descriptor.

        :param dataset_name: Name of the dataset.
        :param dataset_version: Version of the dataset.
        :param start_date: Start date of the dataset to filter the dataset to.
        :param end_date: End date of the dataset to filter the dataset to.
        :param filter_bounds: Geographic bounds for filtering the dataset (west, south, east,
        north).
        :param source_variable_name: Name of the variable in the dataset to be used as source.
        :param target_variable_name: Name of the variable in the dataset to be used as target.
        """
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.start_date = start_date
        self.end_date = end_date
        self.filter_bounds = filter_bounds
        self.source_variable_name = source_variable_name
        self.target_variable_name = target_variable_name

    def __repr__(self) -> str:
        """Return a string representation of the dataset descriptor."""
        return (
            f"NedDatasetDescriptor(dataset_name={self.dataset_name}, "
            f"dataset_version={self.dataset_version}, "
            f"start_date={self.start_date}, "
            f"end_date={self.end_date}, "
            f"filter_bounds={self.filter_bounds}, "
            f"source_variable_name={self.source_variable_name}, "
            f"target_variable_name={self.target_variable_name})"
        )

    @property
    def days_in_range(self) -> int:
        """Calculate the number of days in the date range."""
        return (self.end_date - self.start_date).days + 1
