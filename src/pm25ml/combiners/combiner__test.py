import pytest
from unittest.mock import Mock, MagicMock
from arrow import Arrow
from pm25ml.combiners.combiner import MonthlyCombiner
from pm25ml.collectors.export_pipeline import ExportPipeline, PipelineConfig

TEST_MONTHS = [Arrow(2023, 1, 1), Arrow(2023, 2, 1)]


@pytest.fixture
def mock_combined_storage():
    return Mock()


@pytest.fixture
def mock_archived_wide_combiner():
    return Mock()


@pytest.fixture
def mock_pipeline_config():
    return PipelineConfig(
        id_columns={"date", "grid_id"},
        value_columns={"value1"},
        result_subpath="country=india/dataset=test_dataset/month=2023-01",
        expected_rows=33074 * 31,
    )


@pytest.fixture
def monthly_combiner(mock_combined_storage, mock_archived_wide_combiner):
    return MonthlyCombiner(
        combined_storage=mock_combined_storage,
        archived_wide_combiner=mock_archived_wide_combiner,
        months=TEST_MONTHS,
    )


class MockExportPipeline(ExportPipeline):
    def __init__(self, pipeline_config):
        self._pipeline_config = pipeline_config

    def upload(self) -> None:
        pass

    def get_config_metadata(self):
        return self._pipeline_config


@pytest.fixture
def mock_pipeline_collection():
    static_pipeline = MockExportPipeline(
        PipelineConfig(
            id_columns={"date", "grid_id"},
            value_columns={"value_static"},
            result_subpath="country=india/dataset=test_dataset_static/type=static",
            expected_rows=33074 * 31,
        )
    )
    monthly_pipeline_2023_01 = MockExportPipeline(
        PipelineConfig(
            id_columns={"date", "grid_id"},
            value_columns={"value_monthly"},
            result_subpath="country=india/dataset=test_dataset_monthly/month=2023-01",
            expected_rows=33074 * 31,
        )
    )
    monthly_pipeline_2023_02 = MockExportPipeline(
        PipelineConfig(
            id_columns={"date", "grid_id"},
            value_columns={"value_monthly"},
            result_subpath="country=india/dataset=test_dataset_monthly/month=2023-02",
            expected_rows=33074 * 28,
        )
    )
    year_pipeline = MockExportPipeline(
        PipelineConfig(
            id_columns={"date", "grid_id"},
            value_columns={"value_year"},
            result_subpath="country=india/dataset=test_dataset_yearly/year=2023",
            expected_rows=33074 * 365,
        )
    )

    return [static_pipeline, monthly_pipeline_2023_01, monthly_pipeline_2023_02, year_pipeline]


@pytest.fixture
def mock_successful_result_dataframe_2023_01():
    return Mock(
        shape=(33074 * 31, 3),
        columns=[
            "date",
            "grid_id",
            "test_dataset_static__value_static",
            "test_dataset_monthly__value_monthly",
            "test_dataset_yearly__value_year",
        ],
    )


@pytest.fixture
def mock_successful_result_dataframe_2023_02():
    return Mock(
        shape=(33074 * 28, 3),
        columns=[
            "date",
            "grid_id",
            "test_dataset_static__value_static",
            "test_dataset_monthly__value_monthly",
            "test_dataset_yearly__value_year",
        ],
    )


@pytest.fixture
def mock_combined_storage_with_both_months(
    mock_combined_storage,
    mock_successful_result_dataframe_2023_01,
    mock_successful_result_dataframe_2023_02,
):
    mock_combined_storage.read_dataframe.side_effect = [
        mock_successful_result_dataframe_2023_01,
        mock_successful_result_dataframe_2023_02,
    ]


@pytest.mark.usefixtures(
    "mock_combined_storage_with_both_months",
)
def test__combine_for_months__valid_input__combines_data(
    monthly_combiner,
    mock_archived_wide_combiner,
    mock_pipeline_collection,
    mock_combined_storage_with_both_months,
):
    mock_archived_wide_combiner.needs_combining.return_value = True

    monthly_combiner.combine_for_months(mock_pipeline_collection)

    mock_archived_wide_combiner.combine.assert_any_call(
        month="2023-01",
        processors=mock_pipeline_collection,
    )
    mock_archived_wide_combiner.combine.assert_any_call(
        month="2023-02",
        processors=mock_pipeline_collection,
    )


@pytest.mark.usefixtures(
    "mock_combined_storage_with_both_months",
)
def test__combine_for_months__no_combining_needed__skips_combining(
    monthly_combiner,
    mock_archived_wide_combiner,
    mock_pipeline_collection,
):
    mock_archived_wide_combiner.needs_combining.return_value = False
    monthly_combiner.combine_for_months(mock_pipeline_collection)

    mock_archived_wide_combiner.combine.assert_not_called()


def test__validate_combined__valid_data__passes_validation(
    monthly_combiner, mock_combined_storage, mock_pipeline_collection
):
    month = Arrow(2023, 1, 1)

    mock_combined_storage.read_dataframe.return_value = MagicMock(
        shape=(33074 * 31, 3),
        __getitem__=lambda self, key: self.columns if key == "columns" else None,
        columns=[
            "date",
            "grid_id",
            "test_dataset_static__value_static",
            "test_dataset_monthly__value_monthly",
            "test_dataset_yearly__value_year",
        ],
    )

    monthly_combiner._validate_combined(mock_pipeline_collection, month)


def test__validate_combined__missing_columns__raises_error(
    monthly_combiner, mock_combined_storage, mock_pipeline_collection
):
    month = Arrow(2023, 1, 1)

    mock_combined_storage.read_dataframe.return_value = MagicMock(
        shape=(33074 * 31, 2),
        __getitem__=lambda self, key: self.columns if key == "columns" else None,
        columns=["date", "grid_id"],
    )

    with pytest.raises(ValueError, match="Expected columns"):
        monthly_combiner._validate_combined(mock_pipeline_collection, month)


def test__validate_combined__incorrect_row_count__raises_error(
    monthly_combiner, mock_combined_storage, mock_pipeline_collection
):
    month = Arrow(2023, 1, 1)

    mock_combined_storage.read_dataframe.return_value = MagicMock(
        shape=(33074 * 30, 3),
        __getitem__=lambda self, key: self.columns if key == "columns" else None,
        columns=["date", "grid_id", "test_dataset_static__value_static"],
    )

    with pytest.raises(ValueError, match="Expected 1025294 rows"):
        monthly_combiner._validate_combined(mock_pipeline_collection, month)
