"""Feature planning for gridded feature collections."""

from typing import Any

from arrow import Arrow
from ee.computedobject import ComputedObject
from ee.ee_date import Date
from ee.ee_list import List
from ee.ee_number import Number
from ee.element import Element
from ee.featurecollection import FeatureCollection
from ee.image import Image
from ee.imagecollection import ImageCollection
from ee.reducer import Reducer

from pm25ml.collectors.constants import INDIA_CRS

ISO8601_WITHOUT_TZ = "YYYY-MM-DDTHH:mm:ss"
ISO8601_DATE_ONLY = "YYYY-MM-DD"


class GriddedFeatureCollectionPlanner:
    """
    A planner for creating gridded feature collections from Earth Engine data.

    This class provides methods to process Earth Engine data into gridded feature collections
    for various use cases, such as daily averages, static features, and annual summaries of
    classified pixels.

    Reference: how the "scale" arguments works in Earth Engine: https://developers.google.com/earth-engine/guides/scale
    """

    def __init__(self, grid: FeatureCollection) -> None:
        """
        Initialize the planner with a grid.

        :param grid: The grid to which the features will be mapped.
        :type grid: FeatureCollection
        """
        self.grid = grid

    def plan_daily_average(
        self,
        *,
        collection_name: str,
        selected_bands: list[str],
        dates: list[Arrow],
    ) -> "FeaturePlan":
        """
        Create a daily average feature plan.

        Aggregate pixel-wise mean values for each day and mapping them to the grid.

        :param collection_name: The name of the Earth Engine image collection.
        :type collection_name: str
        :param selected_bands: The bands to include in the aggregation.
        :type selected_bands: list[str]
        :param dates: The list of dates for which daily averages are computed.
        :type dates: list[Arrow]
        :return: A feature plan containing the processed collection and column mappings.
        :rtype: FeaturePlan
        """
        original_raster_scale = self._get_collection_scale(collection_name, selected_bands)

        ids = ["date", "grid_id"]
        transformed_band_names = (
            [f"{band}_mean" for band in selected_bands] if len(selected_bands) > 1 else ["mean"]
        )
        exported_properties = ids + transformed_band_names
        wanted_properties = ids + selected_bands
        column_mappings = dict(zip(exported_properties, wanted_properties))

        # This gets the whole collection and selects the properties we want.
        collection = ImageCollection(collection_name).select(selected_bands)

        date_strings = [date.format(ISO8601_DATE_ONLY) for date in dates]
        gee_dates = List(date_strings)

        def daily_mean_image(date_string: ComputedObject) -> List:
            start = Date(date_string)
            end = start.advance(1, "day")

            return (
                collection.filterDate(
                    start,
                    end,
                )
                .filterBounds(self.grid.geometry())
                # Single value per pixel for the day for each band.
                .reduce(Reducer.mean())
                # We set the date property to the date to carry through
                # to the final export.
                .set("date", start)
            )

        # We create an ImageCollection of daily composites for the month, each
        # the pixel-wise mean value for the day.
        images = ImageCollection.fromImages(
            # This does a server side map operation to create an image for each date.
            gee_dates.map(daily_mean_image),
        )

        # We then average the values for each grid cell for each date.
        def average_grid_value_for_date(im: Image) -> FeatureCollection:
            image_date = im.get("date")

            def carry_date_through(f: Image) -> Element:
                return f.set("date", image_date)

            return im.reduceRegions(
                collection=self.grid,
                # Single value per grid cell for the day.
                reducer=Reducer.mean(),
                crs=INDIA_CRS,
                scale=original_raster_scale,
            ).map(carry_date_through)

        processed_images: FeatureCollection = images.map(average_grid_value_for_date).flatten()

        date_summary = self._common_granularity(dates)

        return FeaturePlan(
            feature_name=self._generate_clean_name(
                "grid-daily-average",
                collection_name,
                date_summary,
            ),
            planned_collection=processed_images,
            column_mappings=column_mappings,
        )

    def plan_static_feature(
        self,
        *,
        image_name: str,
        selected_bands: list[str],
    ) -> "FeaturePlan":
        """
        Create a static feature plan by regridding a single image to the grid.

        :param image_name: The name of the Earth Engine image.
        :type image_name: str
        :param selected_bands: The bands to include in the regridding.
        :type selected_bands: list[str]
        :return: A feature plan containing the processed collection and column mappings.
        :rtype: FeaturePlan
        """
        original_raster_scale = self._get_image_scale(image_name, selected_bands)

        ids = ["grid_id"]
        transformed_band_names = (
            [f"{band}_mean" for band in selected_bands] if len(selected_bands) > 1 else ["mean"]
        )
        exported_properties = ids + transformed_band_names
        wanted_properties = ids + selected_bands
        column_mappings = dict(zip(exported_properties, wanted_properties))

        image = Image(image_name).select(selected_bands)
        collection = ImageCollection.fromImages([image])
        # The only thing we need to do for this is to regrid.
        processed_image: FeatureCollection = collection.map(
            lambda img: img.reduceRegions(
                collection=self.grid,
                reducer=Reducer.mean(),
                crs=INDIA_CRS,
                scale=original_raster_scale,
            ),
        ).flatten()

        return FeaturePlan(
            feature_name=self._generate_clean_name("single-image-grid", image_name),
            planned_collection=processed_image,
            column_mappings=column_mappings,
        )

    def plan_summarise_annual_classified_pixels(
        self,
        *,
        collection_name: str,
        classification_band: str,
        output_names_to_class_values: dict[str, list[int]],
        year: int,
    ) -> "FeaturePlan":
        """
        Create a feature plan for summarizing annual classified pixel data.

        Aggregate by creating boolean bands for each class and taking the mean
        over the year.

        This will give, for each grid cell in the specified year, the percentage
        of time that the pixels in that cell were classified as each of the
        specified classes.

        :param collection_name: The name of the Earth Engine image collection.
        :type collection_name: str
        :param classification_band: The band containing classification data.
        :type classification_band: str
        :param output_names_to_class_values: A mapping of output names to class values.
        :type output_names_to_class_values: dict[str, int]
        :param year: The year for which the summary is computed.
        :type year: int
        :return: A feature plan containing the processed collection and column mappings.
        :rtype: FeaturePlan
        """
        original_raster_scale = self._get_collection_scale(collection_name, [classification_band])

        def add_classes_as_boolean_bands(original_image: Image) -> Image:
            band_column = original_image.select(classification_band)

            new_image = original_image
            for output_name, class_values in output_names_to_class_values.items():
                new_image = new_image.addBands(
                    band_column.remap(
                        class_values,
                        [1] * len(class_values),
                        0,
                        classification_band,
                    ).rename(output_name),
                    [output_name],
                )

            return new_image

        image = Image(
            ImageCollection(collection_name)
            .select(classification_band)
            .filterBounds(self.grid.geometry())
            .filterDate(
                f"{year}-01-01T00:00:00",
                f"{year + 1}-01-01T00:00:00",
            )
            .map(add_classes_as_boolean_bands)
            .select(list(output_names_to_class_values.keys()))
            .reduce(Reducer.mean()),
        )

        # We create an image for the year by filtering the collection
        # and then taking the mean

        collection_for_year = image.reduceRegions(
            collection=self.grid,
            reducer=Reducer.mean(),
            crs=INDIA_CRS,
            scale=original_raster_scale,
        )

        flattened = collection_for_year

        id_columns = ["grid_id"]
        wanted_columns = id_columns + list(output_names_to_class_values.keys())
        exported_columns = id_columns + [
            f"{output_name}_mean" for output_name in output_names_to_class_values
        ]

        column_mappings = dict(zip(exported_columns, wanted_columns))

        return FeaturePlan(
            feature_name=self._generate_clean_name(
                "annual-classified-pixels",
                collection_name,
                year,
            ),
            planned_collection=flattened,
            column_mappings=column_mappings,
            ignore_selectors=True,
        )

    @staticmethod
    def _get_collection_scale(collection_name: str, selected_bands: list[str]) -> Number:
        return (
            ImageCollection(collection_name)
            .select(selected_bands)
            .first()
            .projection()
            .nominalScale()
        )

    @staticmethod
    def _get_image_scale(image_name: str, selected_bands: list[str]) -> Number:
        return Image(image_name).select(selected_bands).projection().nominalScale()

    @staticmethod
    def _generate_clean_name(*args: list[Any]) -> str:
        def clean_name_part(name: str) -> str:
            return name.replace(" ", "-").replace("/", "-").replace("_", "-").lower()

        return "__".join(clean_name_part(str(arg)) for arg in args)

    @staticmethod
    def _common_granularity(dates: list[Arrow]) -> str:
        same_year = all(d.year == dates[0].year for d in dates)
        if not same_year:
            return "x"

        same_month = all(d.month == dates[0].month for d in dates)
        if not same_month:
            return dates[0].format("YYYY")

        same_day = all(d.day == dates[0].day for d in dates)
        if not same_day:
            return dates[0].format("YYYY-MM")

        return dates[0].format("YYYY-MM-DD")


class FeaturePlan:
    """
    Represents a plan for processing and exporting features.

    This class encapsulates the proposed feature collection, column mappings,
    and metadata.
    """

    def __init__(
        self,
        *,
        feature_name: str,
        planned_collection: FeatureCollection,
        column_mappings: dict[str, str],
        ignore_selectors: bool = False,
    ) -> None:
        """
        Initialize a feature plan.

        :param feature_name: The name of the feature plan.
        :type feature_name: str
        :param feature_name: The name of the feature plan.
        :type feature_name: str
        :param planned_collection: The proposed feature collection.
        :type planned_collection: FeatureCollection
        :param column_mappings: A mapping of exported column names to desired column names.
        :type column_mappings: dict[str, str]
        :param ignore_selectors: Whether to ignore selectors during processing. Defaults to False.
        :type ignore_selectors: bool, optional
        """
        self.feature_name = feature_name
        self.planned_collection = planned_collection
        self.column_mappings = column_mappings
        self.ignore_selectors = ignore_selectors

    @property
    def intermediate_columns(self) -> list[str]:
        """
        Return the columns that will be exported to the intermediate storage.

        :return: The keys of the column_mappings dictionary.
        :rtype: list[str]
        """
        return list(self.column_mappings.keys())

    @property
    def wanted_columns(self) -> list[str]:
        """
        Return the columns that are wanted in the final export.

        :return: The values of the column_mappings dictionary.
        :rtype: list[str]
        """
        return list(self.column_mappings.values())
