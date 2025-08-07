"""Controller for imputation using regression models."""

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.recombiner.recombiner import Recombiner
from pm25ml.imputation.from_model.regression_model_imputer import RegressionModelImputer
from pm25ml.logging import logger
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.training.model_pipeline import ModelReference
from pm25ml.training.model_storage import ModelStorage
from pm25ml.training.types import ModelName

IMPUTED_COMBINED_STAGE_NAME = "all_with_imputed"


class RegressionModelImputationController:
    """
    Imputes for each of the project's imputation models.

    Uses the latest available regression model for each and combines to a .
    """

    def __init__(
        self,
        model_store: ModelStorage,
        temporal_config: TemporalConfig,
        combined_storage: CombinedStorage,
        model_refs: dict[ModelName, ModelReference],
        recombiner: Recombiner,
    ) -> None:
        """Build a RegressionModelImputer instance."""
        self.model_store = model_store
        self.temporal_config = temporal_config
        self.combined_storage = combined_storage
        self.model_refs = model_refs
        self.recombiner = recombiner

    def impute(self) -> None:
        """
        Impute the data using the latest regression model for each model.

        Do this for the time period specified by the temporal config.
        """
        to_add_stage_names = []

        for model_name, model_ref in self.model_refs.items():
            logger.info(f"Imputing for model: {model_name}")

            latest_model = self.model_store.load_latest_model(model_name)

            regression_model_imputer = RegressionModelImputer(
                model_ref=model_ref,
                model=latest_model,
                temporal_config=self.temporal_config,
                combined_storage=self.combined_storage,
            )

            regression_model_imputer.impute()
            to_add_stage_names.append(regression_model_imputer.stage_name)

        logger.info("Combining imputed data with generated features")
        self.recombiner.recombine(
            stages=[
                IMPUTED_COMBINED_STAGE_NAME,
                *to_add_stage_names,
            ],
        )
