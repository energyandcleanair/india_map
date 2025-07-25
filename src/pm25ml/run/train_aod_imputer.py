"""Script to train the AOD model."""

from dependency_injector.wiring import Provide, inject

from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env
from pm25ml.training.model_pipeline import ModelPipeline


@inject
def _main(
    model_trainers: dict[str, ModelPipeline] = Provide[Pm25mlContainer.ml_model_trainers],
) -> None:
    aod_trainer = model_trainers["aod"]

    aod_trainer.train_model()


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
