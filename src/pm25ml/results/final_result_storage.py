"""Stores results."""

from typing import BinaryIO

from fsspec import AbstractFileSystem


class FinalResultStorage:
    """Final result storage for various output formats."""

    def __init__(
        self,
        filesystem: AbstractFileSystem,
        destination_bucket: str,
    ) -> None:
        """Initialize the FinalResultStorage."""
        self.filesystem = filesystem
        self.destination_bucket = destination_bucket

    def write(self, data: BinaryIO, path: str, file_name: str) -> None:
        """
        Write the data to the destination bucket.

        :param data: The data to write.
        :param path: The dir path under which to store the data.
        """
        dir_path = f"{self.destination_bucket}/{path}"
        self.filesystem.makedirs(dir_path, exist_ok=True)
        file_path = f"{dir_path}/{file_name}"
        with self.filesystem.open(file_path, "wb") as file:
            file.write(data.read())  # pyright: ignore[reportArgumentType]
            file.flush()
