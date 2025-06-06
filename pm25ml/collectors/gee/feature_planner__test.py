from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from arrow import get

from pm25ml.collectors.gee.feature_planner import (
    FeaturePlan,
    GriddedFeatureCollectionPlanner,
)


def test__FeaturePlan_intermediate_columns__correct_keys__returned() -> None:
    mock_feature_collection = MagicMock()
    column_mappings = {"key1": "value1", "key2": "value2"}
    feature_plan = FeaturePlan(
        feature_name="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
    )

    assert feature_plan.intermediate_columns == ["key1", "key2"]


def test__FeaturePlan_wanted_columns__correct_values__returned() -> None:
    mock_feature_collection = MagicMock()
    column_mappings = {"key1": "value1", "key2": "value2"}
    feature_plan = FeaturePlan(
        feature_name="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
    )

    assert feature_plan.wanted_columns == ["value1", "value2"]

@pytest.fixture
def mock_gee_for_daily_average() -> Iterator[dict[str, MagicMock]]:
    with (
        patch("pm25ml.collectors.gee.feature_planner.ImageCollection") as MockImageCollection,
        patch("pm25ml.collectors.gee.feature_planner.Reducer") as MockReducer,
        patch("pm25ml.collectors.gee.feature_planner.Image") as MockImage,
        patch("pm25ml.collectors.gee.feature_planner.FeatureCollection") as MockFeatureCollection,
        patch("pm25ml.collectors.gee.feature_planner.List") as MockList,
        patch("pm25ml.collectors.gee.feature_planner.Date") as MockDate,
    ):
        mock_ic_instance = MagicMock()
        MockImageCollection.return_value = mock_ic_instance
        mock_ic_instance.select.return_value = mock_ic_instance

        fake_image = MagicMock()
        mock_ic_instance.filterDate.return_value = fake_image
        fake_image.reduce.return_value = fake_image
        fake_image.set.return_value = fake_image
        fake_image.filterBounds.return_value = fake_image

        from_images_mock = MagicMock()
        from_images_mock.map.return_value.flatten.return_value = MagicMock()
        MockImageCollection.fromImages.return_value = from_images_mock

        fake_mean = MagicMock()
        MockReducer.mean.return_value = fake_mean

        mock_list_instance = MagicMock()
        MockList.return_value = mock_list_instance
        mock_list_instance.map.return_value = fake_image

        mock_date_instance_given = MagicMock()
        mock_date_instance_advanced = MagicMock()
        MockDate.return_value = mock_date_instance_given
        mock_date_instance_given.advance.return_value = mock_date_instance_advanced

        yield {
            "MockImageCollection": MockImageCollection,
            "MockReducer": MockReducer,
            "MockImage": MockImage,
            "MockList": MockList,
            "MockFeatureCollection": MockFeatureCollection,
            "mock_ic_instance": mock_ic_instance,
            "fake_image": fake_image,
            "from_images_mock": from_images_mock,
            "fake_mean": fake_mean,
            "mock_list_instance": mock_list_instance,
            "mock_date_instance_given": mock_date_instance_given,
            "mock_date_instance_advanced": mock_date_instance_advanced,
        }


def test__GriddedFeatureCollectionPlanner_plan_daily_average__columns_specified__correct_columns_suggested(
    mock_gee_for_daily_average,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    selected_bands = ["AOD", "PM25"]
    date = get("2023-04-01")

    planner.plan_daily_average(
        collection_name="FAKE/COLLECTION",
        selected_bands=selected_bands,
        dates=[date],
    )

    # Check that we selected the correct bands in all calls
    mock_gee_for_daily_average["mock_ic_instance"].select.assert_called_with(
        selected_bands,
    )


def test__GriddedFeatureCollectionPlanner_plan_daily_average__with_dates__dated_images_processed_correctly(
    mock_gee_for_daily_average,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    dates = [get("2023-01-01"), get("2023-01-02")]

    map_result = MagicMock()
    
    def test_mock_function(func):
        for date in dates:
            func(date)
        return map_result
    mock_gee_for_daily_average["mock_list_instance"].map.side_effect = test_mock_function

    planner.plan_daily_average(
        collection_name="FAKE/COLLECTION",
        selected_bands=["PM25"],
        dates=dates,
    )

    mock_gee_for_daily_average["MockList"].assert_called_once_with(
        [
            "2023-01-01",
            "2023-01-02",
        ],
    )

    # Assert filterDate was called for each date
    # Assert filterDate was called with correct date ranges for each date
    assert mock_gee_for_daily_average["mock_ic_instance"].filterDate.call_count == 2
    assert mock_gee_for_daily_average["mock_ic_instance"].filterDate.call_args_list[0][0] == (
        mock_gee_for_daily_average["mock_date_instance_given"],
        mock_gee_for_daily_average["mock_date_instance_advanced"],
    )

    # Assert reduce and set were called for each date
    assert mock_gee_for_daily_average["fake_image"].set.call_count == 2
    assert mock_gee_for_daily_average["fake_image"].set.call_args_list[0][0][1] == \
        mock_gee_for_daily_average["mock_date_instance_given"]
    assert mock_gee_for_daily_average["fake_image"].set.call_args_list[1][0][1] == \
        mock_gee_for_daily_average["mock_date_instance_given"]

    assert mock_gee_for_daily_average["fake_image"].reduce.call_count == 2

    mock_gee_for_daily_average["MockImageCollection"].fromImages.assert_called_once_with(
        map_result,
    )
    mock_gee_for_daily_average["from_images_mock"].map.assert_called_once()


def test__GriddedFeatureCollectionPlanner_plan_daily_average__gridding__gridding_logic_is_handled_correctly(
    mock_gee_for_daily_average,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    dates = [get("2023-01-01"), get("2023-01-02")]
    planner.plan_daily_average(
        collection_name="FAKE/COLLECTION",
        selected_bands=["PM25"],
        dates=dates,
    )

    # Capture the function passed into the map call
    captured_map_function = mock_gee_for_daily_average["from_images_mock"].map.call_args[0][0]

    # Mock an image with a date property
    mock_image = MagicMock()
    mock_image.get.return_value = "2023-01-01"

    # Mock the feature returned by reduceRegions
    mock_feature = MagicMock()
    mock_feature.set.return_value = mock_feature

    # Mock reduceRegions to return a collection of features
    def mock_map_function(func):
        return [func(mock_feature)]

    mock_image.reduceRegions.return_value.map.side_effect = mock_map_function

    # Call the captured map function directly
    result = captured_map_function(mock_image)

    # Verify that reduceRegions was called with the correct parameters
    mock_image.reduceRegions.assert_called_once_with(
        collection=grid,
        reducer=mock_gee_for_daily_average["MockReducer"].mean(),
        crs="EPSG:7755",  # Updated to match INDIA_CRS
        scale=mock_gee_for_daily_average["mock_ic_instance"].first().projection().nominalScale(),
    )

    # Verify that the date property was set on the feature
    mock_feature.set.assert_called_once_with("date", "2023-01-01")

    # Verify the result
    assert result == [mock_feature]

    mock_gee_for_daily_average["from_images_mock"].map.return_value.flatten.assert_called_once()


@pytest.mark.parametrize(
    ("bands", "expected_column_mappings"),
    [
        (
            ["PM25"],
            {"date": "date", "grid_id": "grid_id", "mean": "PM25"},
        ),
        (
            ["NO2", "SO2"],
            {
                "date": "date",
                "grid_id": "grid_id",
                "NO2_mean": "NO2",
                "SO2_mean": "SO2",
            },
        ),
    ],
    ids=["single_band", "multiple_bands"],
)
@pytest.mark.usefixtures("mock_gee_for_daily_average")
def test__GriddedFeatureCollectionPlanner_plan_daily_average__with_bands__correct_column_mappings_specified(
    bands,
    expected_column_mappings,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    result = planner.plan_daily_average(
        collection_name="ANY",
        selected_bands=bands,
        dates=[get("2023-01-01", "YYYY-MM-DD")],
    )

    assert result.column_mappings == expected_column_mappings


@pytest.fixture
def mock_gee_for_static_feature():
    with (
        patch("pm25ml.collectors.gee.feature_planner.Image") as MockImage,
        patch("pm25ml.collectors.gee.feature_planner.Reducer") as MockReducer,
        patch("pm25ml.collectors.gee.feature_planner.FeatureCollection") as MockFeatureCollection,
        patch("pm25ml.collectors.gee.feature_planner.ImageCollection") as MockImageCollection,
    ):
        mock_image_instance = MagicMock()
        MockImage.return_value = mock_image_instance
        mock_image_instance.select.return_value = mock_image_instance

        fake_mean = MagicMock()
        MockReducer.mean.return_value = fake_mean

        mock_feature_collection_instance = MagicMock()
        mock_image_instance.reduceRegions.return_value = mock_feature_collection_instance

        yield {
            "MockImage": MockImage,
            "MockReducer": MockReducer,
            "MockFeatureCollection": MockFeatureCollection,
            "MockImageCollection": MockImageCollection,
            "mock_image_instance": mock_image_instance,
            "mock_feature_collection_instance": mock_feature_collection_instance,
            "fake_mean": fake_mean,
        }


def test__GriddedFeatureCollectionPlanner_static_feature__columns_specified__correct_columns_suggested(
    mock_gee_for_static_feature,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    selected_bands = ["AOD", "PM25"]

    planner.plan_static_feature(image_name="FAKE/IMAGE", selected_bands=selected_bands)

    # Check that we selected the correct bands
    mock_gee_for_static_feature["mock_image_instance"].select.assert_called_with(
        selected_bands,
    )


def test__GriddedFeatureCollectionPlanner_static_feature__gridding__gridding_logic_is_handled_correctly(
    mock_gee_for_static_feature,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    selected_bands = ["PM25"]

    planner.plan_static_feature(
        image_name="FAKE/IMAGE",
        selected_bands=selected_bands,
    )

    # Verify that reduceRegions was called with the correct parameters
    mock_gee_for_static_feature["mock_image_instance"].reduceRegions.assert_called_once_with(
        collection=grid,
        reducer=mock_gee_for_static_feature["MockReducer"].mean(),
        crs="EPSG:7755",  # Updated to match INDIA_CRS
        scale=mock_gee_for_static_feature["mock_image_instance"].projection().nominalScale(),
    )


@pytest.mark.parametrize(
    ("selected_bands", "expected_column_mappings"),
    [
        (
            ["NO2"],
            {"grid_id": "grid_id", "mean": "NO2"},
        ),
        (
            ["NO2", "SO2"],
            {"grid_id": "grid_id", "NO2_mean": "NO2", "SO2_mean": "SO2"},
        ),
    ],
    ids=["single_band", "multiple_bands"],
)
def test__GriddedFeatureCollectionPlanner_static_feature__column_mappings__correct_mappings_specified(
    mock_gee_for_static_feature,
    selected_bands,
    expected_column_mappings,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)

    result = planner.plan_static_feature(
        image_name="FAKE/IMAGE",
        selected_bands=selected_bands,
    )

    assert result.column_mappings == expected_column_mappings


@pytest.fixture
def mock_gee_for_annual_classified_pixels():
    with (
        patch("pm25ml.collectors.gee.feature_planner.ImageCollection") as MockImageCollection,
        patch("pm25ml.collectors.gee.feature_planner.Reducer") as MockReducer,
        patch("pm25ml.collectors.gee.feature_planner.Image") as MockImage,
        patch("pm25ml.collectors.gee.feature_planner.FeatureCollection") as MockFeatureCollection,
    ):
        mock_ic_instance = MagicMock()
        MockImageCollection.return_value = mock_ic_instance
        mock_ic_instance.select.return_value = mock_ic_instance
        mock_ic_instance.filterBounds.return_value = mock_ic_instance
        mock_ic_instance.filterDate.return_value = mock_ic_instance

        fake_image = MagicMock()
        mock_ic_instance.map.return_value = fake_image
        fake_image.reduce.return_value = fake_image
        fake_image.reduceRegions.return_value = MagicMock()

        fake_mean = MagicMock()
        MockReducer.mean.return_value = fake_mean

        yield {
            "MockImageCollection": MockImageCollection,
            "MockReducer": MockReducer,
            "MockImage": MockImage,
            "MockFeatureCollection": MockFeatureCollection,
            "mock_ic_instance": mock_ic_instance,
            "fake_image": fake_image,
            "fake_mean": fake_mean,
        }


def test__GriddedFeatureCollectionPlanner_annual_classified_pixels__columns_specified__correct_columns_suggested(
    mock_gee_for_annual_classified_pixels,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    classification_band = "land_cover"
    output_names_to_class_values = {"forest": [1], "urban": [2]}
    year = 2023

    planner.plan_summarise_annual_classified_pixels(
        collection_name="FAKE/COLLECTION",
        classification_band=classification_band,
        output_names_to_class_values=output_names_to_class_values,
        year=year,
    )

    # Check that we selected the correct classification band
    mock_gee_for_annual_classified_pixels["mock_ic_instance"].select.assert_called_with(
        classification_band,
    )


def test__GriddedFeatureCollectionPlanner_annual_classified_pixels__gridding__gridding_logic_is_handled_correctly(
    mock_gee_for_annual_classified_pixels,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    classification_band = "land_cover"
    output_names_to_class_values = {"forest": [1], "urban": [2]}
    year = 2023

    planner.plan_summarise_annual_classified_pixels(
        collection_name="FAKE/COLLECTION",
        classification_band=classification_band,
        output_names_to_class_values=output_names_to_class_values,
        year=year,
    )

    # Capture the function passed into the map call
    captured_map_function = mock_gee_for_annual_classified_pixels["mock_ic_instance"].map.call_args[
        0
    ][0]

    # Mock an image with a classification band
    mock_image = MagicMock()
    mock_image.select.return_value.remap.return_value.rename.return_value = mock_image
    mock_image.addBands.return_value = mock_image

    # Call the captured map function directly
    result = captured_map_function(mock_image)

    # Verify that addBands was called with the correct parameters
    mock_image.addBands.assert_called()

    # Verify the result
    assert result == mock_image


@pytest.mark.parametrize(
    ("classification_band", "output_names_to_class_values", "expected_column_mappings"),
    [
        (
            "land_cover",
            {"forest": [1], "urban": [2]},
            {"grid_id": "grid_id", "forest_mean": "forest", "urban_mean": "urban"},
        ),
        (
            "land_use",
            {"agriculture": [3], "water": [4]},
            {
                "grid_id": "grid_id",
                "agriculture_mean": "agriculture",
                "water_mean": "water",
            },
        ),
    ],
    ids=["land_cover", "land_use"],
)
@pytest.mark.usefixtures("mock_gee_for_annual_classified_pixels")
def test__GriddedFeatureCollectionPlanner_annual_classified_pixels__column_mappings__correct_mappings_specified(
    classification_band,
    output_names_to_class_values,
    expected_column_mappings,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)

    result = planner.plan_summarise_annual_classified_pixels(
        collection_name="FAKE/COLLECTION",
        classification_band=classification_band,
        output_names_to_class_values=output_names_to_class_values,
        year=2023,
    )

    assert result.column_mappings == expected_column_mappings
