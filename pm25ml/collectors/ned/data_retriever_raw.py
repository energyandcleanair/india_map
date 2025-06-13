"""Retrieves raw Earth Access data."""

from collections.abc import Iterable
from typing import cast

import earthaccess
from fsspec.spec import AbstractBufferedFile

from pm25ml.collectors.ned.data_retrievers import NedDataRetriever
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.errors import NedMissingDataError
from pm25ml.logging import logger

EARTH_ENGINE_SEARCH_DATE_FORMAT = "YYYY-MM-DD"


class RawEarthAccessDataRetriever(NedDataRetriever):
    """
    Retrieves raw Earth Access data.

    This class streams files from the raw Earth Access source based on the dataset descriptor.
    It searches for granules matching the dataset name and date range, and yields the files
    containing the data for the dataset.

    It does not perform any subsetting or filtering of the data before yielding the files.
    """

    def stream_files(
        self,
        *,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> Iterable[AbstractBufferedFile]:
        """
        Stream data from the raw Earth Access source.

        Args:
            dataset_descriptor (NedDatasetDescriptor): The dataset descriptor containing metadata
                and processing instructions.

        Returns:
            Iterable[AbstractBufferedFile]: An iterable of files containing the data for the
            dataset.

        """
        logger.info("Searching for granules for dataset %s", dataset_descriptor)
        granules: list[earthaccess.DataGranule] = earthaccess.search_data(
            short_name=dataset_descriptor.dataset_name,
            temporal=(
                dataset_descriptor.start_date.format(EARTH_ENGINE_SEARCH_DATE_FORMAT),
                dataset_descriptor.end_date.format(EARTH_ENGINE_SEARCH_DATE_FORMAT),
            ),
            count=-1,
            version=dataset_descriptor.dataset_version,
        )

        if len(granules) == 0:
            msg = f"No granules found for dataset {dataset_descriptor}."
            raise NedMissingDataError(msg)

        expected_days = dataset_descriptor.days_in_range
        if len(granules) != expected_days:
            msg = (
                f"Expected {expected_days} granules for dataset {dataset_descriptor}, "
                f"but found {len(granules)}."
            )
            raise NedMissingDataError(
                msg,
            )

        logger.info(
            "Found %d granules for dataset %s",
            len(granules),
            dataset_descriptor,
        )

        for granule in granules:
            [file] = earthaccess.open([granule])
            # earthaccess misreports the return type of the file as an AbstractFileSystem
            # but it is actually an AbstractBufferedFile.
            yield cast("AbstractBufferedFile", file)
