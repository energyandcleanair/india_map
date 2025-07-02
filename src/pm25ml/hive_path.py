"""A module to handle paths with metadata extraction for Hive results."""


class HivePath:
    """
    A class to represent a path with metadata.

    This class is used to extract metadata from the result subpath.
    """

    def __init__(self, result_subpath: str) -> None:
        """
        Initialize the PathWithMetadata with the result subpath.

        :param result_subpath: The subpath where the result is stored.
        """
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
