import pytest
from unittest.mock import Mock
from pm25ml.training.model_pipeline import ModelReference
from pm25ml.training.types import ModelName, Pm25mlCompatibleModel


@pytest.fixture
def mock_model_name_fixture():
    return Mock(spec=ModelName)


@pytest.fixture
def mock_model_builder_fixture() -> Pm25mlCompatibleModel:
    mock_model = Mock()
    return mock_model


@pytest.fixture
def dummy_extra_sampler_fixture():
    def sampler(lazy_frame):
        return lazy_frame

    return sampler


def test__model_reference__target_in_predictor_cols__raises_value_error(
    mock_model_name_fixture, mock_model_builder_fixture, dummy_extra_sampler_fixture
):
    with pytest.raises(ValueError, match="Target column 'target' cannot be in predictor columns"):
        ModelReference(
            source_stage=mock_model_name_fixture,  # Use fixture for ModelName
            predictor_cols=["target", "feature1", "feature2"],
            target_col="target",
            grouper_col="group",
            model_builder=mock_model_builder_fixture,
            extra_sampler=dummy_extra_sampler_fixture,
        )


def test__model_reference__grouper_in_predictor_cols__raises_value_error(
    mock_model_name_fixture, mock_model_builder_fixture, dummy_extra_sampler_fixture
):
    with pytest.raises(ValueError, match="Grouper column 'group' cannot be in predictor columns"):
        ModelReference(
            source_stage=mock_model_name_fixture,  # Use fixture for ModelName
            predictor_cols=["group", "feature1", "feature2"],
            target_col="target",
            grouper_col="group",
            model_builder=mock_model_builder_fixture,
            extra_sampler=dummy_extra_sampler_fixture,
        )


def test__model_reference__valid_columns__does_not_raise_exception(
    mock_model_name_fixture, mock_model_builder_fixture, dummy_extra_sampler_fixture
):
    try:
        ModelReference(
            source_stage=mock_model_name_fixture,  # Use fixture for ModelName
            predictor_cols=["feature1", "feature2"],
            target_col="target",
            grouper_col="group",
            model_builder=mock_model_builder_fixture,
            extra_sampler=dummy_extra_sampler_fixture,
        )
    except ValueError:
        pytest.fail("ModelReference raised ValueError unexpectedly!")
