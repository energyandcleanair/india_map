"""A module to handle paths with metadata extraction for Hive results."""


class HivePath:
    """
    A class to represent a path with metadata.

    This class is used to extract metadata from the result subpath.
    """

    @staticmethod
    def from_args(**kwargs: str) -> "HivePath":
        """
        Create a HivePath from kwargs.

        This will construct the path in the order that the args are passed in.

        :param kwargs: Ordered key-value pairs representing the metadata.
        :return: A HivePath instance.
        """
        result_subpath = "/".join(f"{key}={value}" for key, value in kwargs.items())
        return HivePath(result_subpath=result_subpath)

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

    def __eq__(self, value: object) -> bool:
        """
        Check if the value is equal to this HivePath instance.

        :param value: The value to compare with.
        :return: True if the value is a HivePath and has the same result subpath, False otherwise.
        """
        return isinstance(value, HivePath) and self.result_subpath == value.result_subpath

    def __str__(self) -> str:
        """
        Return a string representation of the HivePath.

        :return: The result subpath.
        """
        return self.result_subpath

    def __repr__(self) -> str:
        """
        Return a string representation of the HivePath for debugging.

        :return: A string representation of the HivePath.
        """
        return f"HivePath(result_subpath={self.result_subpath!r})"

    def __hash__(self) -> int:
        """
        Return a hash of the HivePath instance.

        :return: The hash of the result subpath.
        """
        return hash(self.result_subpath)
