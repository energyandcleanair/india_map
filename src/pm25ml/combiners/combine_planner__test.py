from arrow import Arrow
from pm25ml.collectors.validate_configuration import VALID_COUNTRIES
from pm25ml.combiners.combine_planner import CombinePlan


def test__CombinePlan__month_id():
    desc = CombinePlan(
        month=Arrow(2023, 1, 1),
        paths=[],
        expected_columns=["col1", "col2"],
    )

    assert desc.month_id == "2023-01", "Month ID should be formatted as 'YYYY-MM'"


def test__CombinePlan__expected_rows():
    desc = CombinePlan(
        month=Arrow(2023, 1, 1),
        paths=[],
        expected_columns=["col1", "col2"],
    )

    assert desc.expected_rows == VALID_COUNTRIES["india"] * 31, (
        "Expected rows should be equal to the number of days in the month"
    )


def test__CombinePlan__days_in_month():
    desc = CombinePlan(
        month=Arrow(2023, 1, 1),
        paths=[],
        expected_columns=["col1", "col2"],
    )

    assert desc.days_in_month == 31, "Days in month should be 31 for January"
