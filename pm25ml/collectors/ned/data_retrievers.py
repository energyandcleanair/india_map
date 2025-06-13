"""Data retriever for NASA Earthdata sources."""

from collections.abc import Iterable

from fsspec.spec import AbstractBufferedFile

from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor

EARTH_ENGINE_SEARCH_DATE_FORMAT = "YYYY-MM-DD"
HARMONY_DATE_FILTER_FORMAT = "YYYY-MM-DDTHH:mm:ssZ"


class NedDataRetriever:
    """
    Retrieves data from a NASA source.

    Streams files from the source based on the dataset descriptor. This may
    involve searching for datasets, granules, and initiating subsetting jobs.
    """

    def stream_files(
        self,
        *,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> Iterable[AbstractBufferedFile]:
        """Stream files from the source."""
        msg = "This method should be implemented by subclasses."
        raise NotImplementedError(msg)
