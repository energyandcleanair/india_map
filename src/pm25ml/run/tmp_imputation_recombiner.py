"""Script to train the full model."""

from dependency_injector.wiring import Provide, inject

from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.combiners.recombiner.recombiner import Recombiner
from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env


@inject
def _main(
    imputation_recombiner: Recombiner = Provide[Pm25mlContainer.imputer_recombiner],
    imputed_data_artifact: DataArtifactRef = Provide[
        Pm25mlContainer.data_artifacts_container.ml_imputed_super_stage
    ],
) -> None:
    imputation_recombiner.recombine(
        stages=[
            imputed_data_artifact.for_sub_artifact("aod"),
            imputed_data_artifact.for_sub_artifact("no2"),
            imputed_data_artifact.for_sub_artifact("co"),
        ],
    )


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
