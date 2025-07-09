from polars import DataFrame
import pytest
from unittest.mock import Mock, MagicMock
from arrow import Arrow
from pm25ml.combiners.combine_planner import CombinePlan
from pm25ml.combiners.combiner import MonthlyCombiner
from pm25ml.collectors.export_pipeline import ExportPipeline, PipelineConfig
from pm25ml.hive_path import HivePath

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
    )


@pytest.fixture
def combine_defs():
    return [
        CombinePlan(
            month=Arrow(2023, 1, 1),
            paths=[
                HivePath("country=india/dataset=test_dataset_static/type=static"),
                HivePath("country=india/dataset=test_dataset_monthly/month=2023-01"),
                HivePath("country=india/dataset=test_dataset_yearly/year=2023"),
            ],
            expected_columns=[
                "date",
                "grid_id",
                "test_dataset_static__value_static",
                "test_dataset_monthly__value_monthly",
                "test_dataset_yearly__value_year",
            ],
        ),
        CombinePlan(
            month=Arrow(2023, 2, 1),
            paths=[
                HivePath("country=india/dataset=test_dataset_static/type=static"),
                HivePath("country=india/dataset=test_dataset_monthly/month=2023-02"),
                HivePath("country=india/dataset=test_dataset_yearly/year=2023"),
            ],
            expected_columns=[
                "date",
                "grid_id",
                "test_dataset_static__value_static",
                "test_dataset_monthly__value_monthly",
                "test_dataset_yearly__value_year",
            ],
        ),
    ]


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
    combine_defs,
):
    mock_archived_wide_combiner.needs_combining.return_value = True

    monthly_combiner.combine_for_months(combine_defs)

    mock_archived_wide_combiner.combine.assert_any_call(
        combine_defs[0],
    )
    mock_archived_wide_combiner.combine.assert_any_call(
        combine_defs[1],
    )


@pytest.mark.usefixtures(
    "mock_combined_storage_with_both_months",
)
def test__combine_for_months__no_combining_needed__skips_combining(
    monthly_combiner,
    mock_archived_wide_combiner,
    combine_defs,
):
    mock_archived_wide_combiner.needs_combining.return_value = False
    monthly_combiner.combine_for_months(combine_defs)

    mock_archived_wide_combiner.combine.assert_not_called()


def test__validate_combined__valid_data__passes_validation(
    monthly_combiner, mock_combined_storage, combine_defs
):
    mock_combined_storage.read_dataframe.side_effect = [
        DataFrame(
            {
                "date": ["2023-01-01"] * (33074 * 31),
                "grid_id": list(range(33074 * 31)),
                "test_dataset_static__value_static": [1.0] * (33074 * 31),
                "test_dataset_monthly__value_monthly": [2.0] * (33074 * 31),
                "test_dataset_yearly__value_year": [3.0] * (33074 * 31),
            }
        ),
        DataFrame(
            {
                "date": ["2023-02-01"] * (33074 * 28),
                "grid_id": list(range(33074 * 28)),
                "test_dataset_static__value_static": [1.0] * (33074 * 28),
                "test_dataset_monthly__value_monthly": [2.0] * (33074 * 28),
                "test_dataset_yearly__value_year": [3.0] * (33074 * 28),
            }
        ),
    ]

    monthly_combiner.combine_for_months(combine_defs)


def test__validate_combined__missing_columns__raises_error(
    monthly_combiner, mock_combined_storage, combine_defs
):
    mock_combined_storage.read_dataframe.side_effect = [
        DataFrame(
            {
                "date": ["2023-01-01"] * (33074 * 31),
                "grid_id": list(range(33074 * 31)),
                "test_dataset_static__value_static": [1.0] * (33074 * 31),
            }
        ),
        DataFrame(
            {
                "date": ["2023-02-01"] * (33074 * 28),
                "grid_id": list(range(33074 * 28)),
                "test_dataset_static__value_static": [1.0] * (33074 * 28),
            }
        ),
    ]

    with pytest.raises(ValueError, match="Expected columns"):
        monthly_combiner.combine_for_months(combine_defs)


def test__validate_combined__incorrect_row_count__raises_error(
    monthly_combiner, mock_combined_storage, combine_defs
):
    mock_combined_storage.read_dataframe.side_effect = [
        DataFrame(
            {
                "date": ["2023-01-01"] * (33074 * 30),
                "grid_id": list(range(33074 * 30)),
                "test_dataset_static__value_static": [1.0] * (33074 * 30),
                "test_dataset_monthly__value_monthly": [2.0] * (33074 * 30),
                "test_dataset_yearly__value_year": [3.0] * (33074 * 30),
            }
        ),
        DataFrame(
            {
                "date": ["2023-02-01"] * (33074 * 28),
                "grid_id": list(range(33074 * 28)),
                "test_dataset_static__value_static": [1.0] * (33074 * 28),
                "test_dataset_monthly__value_monthly": [2.0] * (33074 * 28),
                "test_dataset_yearly__value_year": [3.0] * (33074 * 28),
            }
        ),
    ]

    with pytest.raises(ValueError, match="Expected 1025294 rows"):
        monthly_combiner.combine_for_months(combine_defs)
