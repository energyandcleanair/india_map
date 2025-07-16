import numpy as np
import polars as pl
from pm25ml.imputation.spatial.daily_spatial_interpolator import DailySpatialInterpolator
from pm25ml.collectors.grid_loader import Grid
from polars.testing import assert_frame_equal


def test__impute__single_day_single_column__correct_imputation():
    # Create a 4x4 grid of points
    grid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    x_coords = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    y_coords = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

    grid_data = pl.DataFrame(
        {
            "grid_id": grid_ids,
            Grid.ORIGINAL_GEOM_COL: [f"POINT({x} {y})" for x, y in zip(x_coords, y_coords)],
            Grid.ORIGINAL_X: x_coords,
            Grid.ORIGINAL_Y: y_coords,
            Grid.GEOM_COL: [None] * 16,
            Grid.LAT_COL: [None] * 16,
            Grid.LON_COL: [None] * 16,
        }
    )
    grid = Grid(df=grid_data)

    original_data = np.array(
        [
            1.0,
            2.0,
            np.nan,
            4.0,
            np.nan,
            6.0,
            7.0,
            np.nan,
            9.0,
            np.nan,
            11.0,
            12.0,
            np.nan,
            14.0,
            15.0,
            np.nan,
        ]
    )

    # Create input data with some missing values
    input_data = pl.DataFrame(
        {
            "grid_id": grid_ids,
            "date": ["2025-07-15"] * 16,
            "month": ["2025-07"] * 16,
            "value": original_data,
        }
    )

    # Expected output data:
    # 2, 0: calculated linearly across the x-axis
    # 0, 1: calculated linearly across the y-axis
    # 3, 1: calculated linearly across the y-axis
    # 1, 2: calculated linearly across the x and y axes
    # 3, 0: uses nn from the nearest point, taking the first available value
    # 3, 3: uses nn from the nearest point, taking the first available value
    # Remaining points aren't changed
    expected_data = pl.DataFrame(
        {
            "grid_id": grid_ids,
            "date": ["2025-07-15"] * 16,
            "month": ["2025-07"] * 16,
            "value": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                9.0,
                14.0,
                15.0,
                12.0,
            ],
        }
    )

    # Initialize the imputer
    imputer = DailySpatialInterpolator(grid=grid, value_column_regex_selector=r"^value$")

    # Perform imputation
    result = imputer.impute(input_data)

    # Assert the result matches the expected output
    assert_frame_equal(
        result,
        expected_data,
        check_column_order=True,
        check_row_order=True,
    )


def test__impute__multiple_dates__separate_imputation():
    # Create a 3x3 grid of points
    grid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_coords = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_coords = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    grid_data = pl.DataFrame(
        {
            "grid_id": grid_ids,
            Grid.ORIGINAL_GEOM_COL: [f"POINT({x} {y})" for x, y in zip(x_coords, y_coords)],
            Grid.ORIGINAL_X: x_coords,
            Grid.ORIGINAL_Y: y_coords,
            Grid.GEOM_COL: [None] * 9,
            Grid.LAT_COL: [None] * 9,
            Grid.LON_COL: [None] * 9,
        }
    )
    grid = Grid(df=grid_data)

    # Create input data with some missing values for two dates
    input_data = pl.DataFrame(
        {
            "grid_id": grid_ids * 2,
            "date": ["2025-07-15"] * 9 + ["2025-07-16"] * 9,
            "month": ["2025-07"] * 18,
            "value": [
                1.0,
                2.0,
                3.0,
                4.0,
                np.nan,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                20.0,
                30.0,
                40.0,
                np.nan,
                60.0,
                70.0,
                80.0,
                90.0,
            ],
        }
    )

    # Expected output data: missing value (5, 5) is calculated separately for each date
    expected_data = pl.DataFrame(
        {
            "grid_id": grid_ids * 2,
            "date": ["2025-07-15"] * 9 + ["2025-07-16"] * 9,
            "month": ["2025-07"] * 18,
            "value": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
            ],
        }
    )

    # Initialize the imputer
    imputer = DailySpatialInterpolator(grid=grid, value_column_regex_selector=r"^value$")

    # Perform imputation
    result = imputer.impute(input_data)

    # Assert the result matches the expected output
    assert_frame_equal(
        result,
        expected_data,
        check_column_order=True,
        check_row_order=True,
    )


def test__impute__multiple_columns__regex_selection():
    # Create a 3x3 grid of points
    grid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_coords = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_coords = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    grid_data = pl.DataFrame(
        {
            "grid_id": grid_ids,
            Grid.ORIGINAL_GEOM_COL: [f"POINT({x} {y})" for x, y in zip(x_coords, y_coords)],
            Grid.ORIGINAL_X: x_coords,
            Grid.ORIGINAL_Y: y_coords,
            Grid.GEOM_COL: [None] * 9,
            Grid.LAT_COL: [None] * 9,
            Grid.LON_COL: [None] * 9,
        }
    )
    grid = Grid(df=grid_data)

    # Create input data with multiple value columns and some missing values
    input_data = pl.DataFrame(
        {
            "grid_id": grid_ids,
            "date": ["2025-07-15"] * 9,
            "month": ["2025-07"] * 9,
            "value_1": [
                1.0,
                2.0,
                3.0,
                4.0,
                np.nan,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
            "value_2": [
                10.0,
                20.0,
                30.0,
                40.0,
                np.nan,
                60.0,
                70.0,
                80.0,
                90.0,
            ],
            "ignored_column": [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
            ],
        }
    )

    # Expected output data: missing values are calculated separately for each selected column
    expected_data = pl.DataFrame(
        {
            "grid_id": grid_ids,
            "date": ["2025-07-15"] * 9,
            "month": ["2025-07"] * 9,
            "value_1": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
            "value_2": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
            ],
            "ignored_column": [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
            ],
        }
    )

    # Initialize the imputer with a regex to select value columns
    imputer = DailySpatialInterpolator(grid=grid, value_column_regex_selector=r"^value_\d$")

    # Perform imputation
    result = imputer.impute(input_data)

    # Assert the result matches the expected output
    assert_frame_equal(
        result,
        expected_data,
        check_column_order=True,
        check_row_order=True,
    )
