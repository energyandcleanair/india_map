"""Retrieves raw Earth Access data."""

from collections.abc import Iterable
from typing import cast

import earthaccess
from fsspec.spec import AbstractBufferedFile

from pm25ml.collectors.ned.data_retrievers import EARTH_ENGINE_SEARCH_DATE_FORMAT, NedDataRetriever
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.errors import NedMissingDataError
from pm25ml.logging import logger


class RawEarthAccessDataRetriever(NedDataRetriever):
    """Retrieves raw Earth Access data."""

    def stream_files(
        self,
        *,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> Iterable[AbstractBufferedFile]:
        """Stream data from the raw Earth Access source."""
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
            yield cast("AbstractBufferedFile", file)
