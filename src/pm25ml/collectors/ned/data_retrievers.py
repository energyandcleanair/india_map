"""Data retriever for NASA Earthdata sources."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import IO

from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor


class NedDataRetriever(ABC):
    """
    Retrieves data from a NASA source.

    Streams files from the source based on the dataset descriptor. Optionally, this may allow for
    server-side subsetting operations based on the dataset descriptor.
    """

    @abstractmethod
    def stream_files(
        self,
        *,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> Iterable[IO[bytes]]:
        """
        Stream files from the source.

        Optionally, this may allow for server-side subsetting operations based on the
        dataset descriptor.

        Args:
            dataset_descriptor (NedDatasetDescriptor): The dataset descriptor containing metadata
                and processing instructions.

        Returns:
            Iterable[IO[bytes]]: An iterable of files containing the data for the
            dataset.

        """
