"""Runner to get the data from a variety of sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

from pm25ml.collectors.validate_configuration import validate_configuration
from pm25ml.logging import logger
from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env

if TYPE_CHECKING:
    from collections.abc import Collection

    from pm25ml.collectors.collector import RawDataCollector
    from pm25ml.collectors.export_pipeline import (
        ExportPipeline,
    )
    from pm25ml.combiners.combiner import MonthlyCombiner


@inject
def _main(
    processors: Collection[ExportPipeline] = Provide[Pm25mlContainer.pipelines],
    collector: RawDataCollector = Provide[Pm25mlContainer.collector],
    monthly_combiner: MonthlyCombiner = Provide[Pm25mlContainer.monthly_combiner],
) -> None:
    logger.info("Validating export pipeline config")
    validate_configuration(processors)

    logger.info("Combining results from the archive storage")
    collector.collect(processors)

    # Get files from the archive storage
    logger.info("Combining results from the archive storage")
    monthly_combiner.combine_for_months(processors)


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
