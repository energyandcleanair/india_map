import pytest
from unittest.mock import patch, MagicMock
from arrow import get

from pm25ml.collectors.feature_planner import FeatureCollectionPlanner, FeaturePlan


@pytest.fixture
def mock_gee():
    with patch("pm25ml.collectors.feature_planner.ImageCollection") as MockImageCollection, patch(
        "pm25ml.collectors.feature_planner.Reducer"
    ) as MockReducer, patch(
        "pm25ml.collectors.feature_planner.Image"
    ) as MockImage, patch(
        "pm25ml.collectors.feature_planner.FeatureCollection"
    ) as MockFeatureCollection:

        mock_ic_instance = MagicMock()
        MockImageCollection.return_value = mock_ic_instance
        mock_ic_instance.select.return_value = mock_ic_instance

        fake_image = MagicMock()
        mock_ic_instance.filterDate.return_value = fake_image
        fake_image.reduce.return_value = fake_image
        fake_image.set.return_value = fake_image

        from_images_mock = MagicMock()
        from_images_mock.map.return_value.flatten.return_value = MagicMock()
        MockImageCollection.fromImages.return_value = from_images_mock

        fake_mean = MagicMock()
        MockReducer.mean.return_value = fake_mean

        yield {
            "MockImageCollection": MockImageCollection,
            "MockReducer": MockReducer,
            "MockImage": MockImage,
            "MockFeatureCollection": MockFeatureCollection,
            "mock_ic_instance": mock_ic_instance,
            "fake_image": fake_image,
            "from_images_mock": from_images_mock,
            "fake_mean": fake_mean,
        }

def test_FeatureCollectionPlanner_columnsSpecified_correctColumnsSuggested(mock_gee):
    grid = MagicMock()
    planner = FeatureCollectionPlanner(grid=grid)
    selected_bands = ["AOD", "PM25"]
    date = get("2023-04-01")

    planner.plan_grid_daily_average(
        collection_name="FAKE/COLLECTION", selected_bands=selected_bands, dates=[date]
    )

    # Check that we selected the correct bands
    mock_gee["mock_ic_instance"].select.assert_called_once_with(selected_bands)

def test_FeatureCollectionPlanner_withDates_datedImagesProcessedCorrectly(mock_gee):
    grid = MagicMock()
    planner = FeatureCollectionPlanner(grid=grid)
    dates = [get("2023-01-01"), get("2023-01-02")]

    planner.plan_grid_daily_average(
        collection_name="FAKE/COLLECTION",
        selected_bands=["PM25"],
        dates=dates,
    )

    # Assert filterDate was called for each date
    # Assert filterDate was called with correct date ranges for each date
    assert mock_gee["mock_ic_instance"].filterDate.call_count == 2
    assert mock_gee["mock_ic_instance"].filterDate.call_args_list[0][0] == ("2023-01-01T00:00:00", "2023-01-02T00:00:00")
    assert mock_gee["mock_ic_instance"].filterDate.call_args_list[1][0] == ("2023-01-02T00:00:00", "2023-01-03T00:00:00")

    # Assert reduce and set were called for each date
    assert mock_gee["fake_image"].set.call_count == 2
    assert mock_gee["fake_image"].set.call_args_list[0][0][1] == "2023-01-01"
    assert mock_gee["fake_image"].set.call_args_list[1][0][1] == "2023-01-02"
    
    assert mock_gee["fake_image"].reduce.call_count == 2

    mock_gee["MockImageCollection"].fromImages.assert_called_once_with([
        mock_gee["fake_image"], mock_gee["fake_image"]
    ])
    mock_gee["from_images_mock"].map.assert_called_once()

def test_FeatureCollectionPlanner_gridding_griddingLogicIsHandledCorrectly(mock_gee):

    grid = MagicMock()
    planner = FeatureCollectionPlanner(grid=grid)
    dates = [get("2023-01-01"), get("2023-01-02")]
    planner.plan_grid_daily_average(
        collection_name="FAKE/COLLECTION",
        selected_bands=["PM25"],
        dates=dates,
    )

    # Capture the function passed into the map call
    captured_map_function = mock_gee["from_images_mock"].map.call_args[0][0]

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
        reducer=mock_gee["MockReducer"].mean(),
        crs="EPSG:7755",  # Updated to match INDIA_CRS
        scale=10000,  # Assuming SCALE_10KM is 10000
    )

    # Verify that the date property was set on the feature
    mock_feature.set.assert_called_once_with("date", "2023-01-01")

    # Verify the result
    assert result == [mock_feature]

    mock_gee["from_images_mock"].map.return_value.flatten.assert_called_once()


@pytest.mark.parametrize(
    "bands,expected_column_mappings",
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
@pytest.mark.usefixtures("mock_gee")
def test_FeatureCollectionPlanner_withBands_correctColumnMappingsSpecified(bands, expected_column_mappings):
    grid = MagicMock()
    planner = FeatureCollectionPlanner(grid=grid)
    result = planner.plan_grid_daily_average(
        collection_name="ANY",
        selected_bands=bands,
        dates=[get("2023-01-01", "YYYY-MM-DD")],
    )

    assert result.column_mappings == expected_column_mappings


def test_FeaturePlan_intermediateColumns_correctKeysReturned():
    mock_feature_collection = MagicMock()
    column_mappings = {"key1": "value1", "key2": "value2"}
    feature_plan = FeaturePlan(
        type="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
    )

    assert feature_plan.intermediate_columns == ["key1", "key2"]

def test_FeaturePlan_wantedColumns_correctValuesReturned():
    mock_feature_collection = MagicMock()
    column_mappings = {"key1": "value1", "key2": "value2"}
    feature_plan = FeaturePlan(
        type="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
    )

    assert feature_plan.wanted_columns == ["value1", "value2"]

