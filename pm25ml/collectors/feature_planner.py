from ee import ImageCollection, Reducer, FeatureCollection, Image
from arrow import Arrow

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

    def __init__(self, grid: FeatureCollection):
        """
        Initializes the planner with a grid.

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
        Creates a daily average feature plan by aggregating pixel-wise mean values for each day
        and mapping them to the grid.

        :param collection_name: The name of the Earth Engine image collection.
        :type collection_name: str
        :param selected_bands: The bands to include in the aggregation.
        :type selected_bands: list[str]
        :param dates: The list of dates for which daily averages are computed.
        :type dates: list[Arrow]
        :return: A feature plan containing the processed collection and column mappings.
        :rtype: FeaturePlan
        """

        original_raster_scale = self._get_collection_scale(collection_name)

        ids = ["date", "grid_id"]
        transformed_band_names = (
            [f"{band}_mean" for band in selected_bands] if len(selected_bands) > 1 else ["mean"]
        )
        exported_properties = ids + transformed_band_names
        wanted_properties = ids + selected_bands
        column_mappings = {
            exported: wanted for exported, wanted in zip(exported_properties, wanted_properties)
        }

        # This gets the whole collection and selects the properties we want.
        collection = ImageCollection(collection_name).select(selected_bands)

        # We create an ImageCollection of daily composites for the month, each
        # the pixel-wise mean value for the day.
        images = ImageCollection.fromImages(
            [
                collection.filterDate(
                    date.format(ISO8601_WITHOUT_TZ),
                    date.shift(days=1).format(ISO8601_WITHOUT_TZ),
                )
                # Single value per pixel for the day for each band.
                .reduce(Reducer.mean())
                # We set the date property to the date to carry through
                # to the final export.
                .set("date", date.format(ISO8601_DATE_ONLY))
                for date in dates
            ]
        )

        # We then average the values for each grid cell for each date.
        def average_grid_value_for_date(im: Image):
            image_date = im.get("date")

            def carry_date_through(f: Image) -> Image:
                return f.set("date", image_date)

            return im.reduceRegions(
                collection=self.grid,
                # Single value per grid cell for the day.
                reducer=Reducer.mean(),
                crs=INDIA_CRS,
                scale=original_raster_scale,
            ).map(carry_date_through)

        processed_images: FeatureCollection = images.map(average_grid_value_for_date).flatten()

        return FeaturePlan(
            type="grid-daily-average",
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
        Creates a static feature plan by regridding a single image to the grid.

        :param image_name: The name of the Earth Engine image.
        :type image_name: str
        :param selected_bands: The bands to include in the regridding.
        :type selected_bands: list[str]
        :return: A feature plan containing the processed collection and column mappings.
        :rtype: FeaturePlan
        """
        original_raster_scale = self._get_image_scale(image_name)

        ids = ["grid_id"]
        transformed_band_names = (
            [f"{band}_mean" for band in selected_bands] if len(selected_bands) > 1 else ["mean"]
        )
        exported_properties = ids + transformed_band_names
        wanted_properties = ids + selected_bands
        column_mappings = {
            exported: wanted for exported, wanted in zip(exported_properties, wanted_properties)
        }

        image = Image(image_name).select(selected_bands)
        # The only thing we need to do for this is to regrid.
        processed_image: FeatureCollection = image.reduceRegions(
            collection=self.grid,
            reducer=Reducer.mean(),
            crs=INDIA_CRS,
            scale=original_raster_scale,
        )
        return FeaturePlan(
            type="single-image-grid",
            planned_collection=processed_image,
            column_mappings=column_mappings,
        )

    def plan_summarise_annual_classified_pixels(
        self,
        *,
        collection_name: str,
        classification_band: str,
        output_names_to_class_values: dict[str, int],
        year: int,
    ) -> "FeaturePlan":
        """
        Creates a feature plan for summarizing annual classified pixel data by
        creating boolean bands for each class and aggregating them over the year.

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

        original_raster_scale = self._get_collection_scale(collection_name)

        def add_classes_as_boolean_bands(original_image: Image) -> Image:
            band_column = original_image.select(classification_band)

            new_image = original_image
            for output_name, class_values in output_names_to_class_values.items():
                new_image = new_image.addBands(
                    band_column.remap(
                        class_values, [1] * len(class_values), 0, classification_band
                    ).rename(output_name),
                    [output_name],
                )

            return new_image

        with_pivoted_columns = (
            ImageCollection(collection_name)
            .select(classification_band)
            .map(add_classes_as_boolean_bands)
            .select(list(output_names_to_class_values.keys()))
        )

        # We create an image for the year by filtering the collection
        # and then taking the mean
        image_for_year = Image(
            with_pivoted_columns.filterDate(
                f"{year}-01-01T00:00:00", f"{year + 1}-01-01T00:00:00"
            ).reduce(Reducer.mean())
        )

        collection_for_year = image_for_year.reduceRegions(
            collection=self.grid,
            reducer=Reducer.mean(),
            crs=INDIA_CRS,
            scale=original_raster_scale,
        )

        flattened = collection_for_year.flatten()

        id_columns = ["grid_id"]
        wanted_columns = id_columns + list(output_names_to_class_values.keys())
        exported_columns = id_columns + [
            f"{output_name}_mean" for output_name in output_names_to_class_values.keys()
        ]

        column_mappings = {
            exported: wanted for exported, wanted in zip(exported_columns, wanted_columns)
        }

        return FeaturePlan(
            type="annual-classified-pixels",
            planned_collection=flattened,
            column_mappings=column_mappings,
            ignore_selectors=True,
        )

    @staticmethod
    def _get_collection_scale(collection_name: str):
        """
        Retrieves the nominal scale of the first image in the specified collection.

        :param collection_name: The name of the Earth Engine image collection.
        :type collection_name: str
        :return: The nominal scale of the collection.
        :rtype: float
        """
        return ImageCollection(collection_name).first().projection().nominalScale()

    @staticmethod
    def _get_image_scale(image_name: str):
        """
        Retrieves the nominal scale of the specified image.

        :param image_name: The name of the Earth Engine image.
        :type image_name: str
        :return: The nominal scale of the image.
        :rtype: float
        """
        return Image(image_name).projection().nominalScale()


class FeaturePlan:
    """
    Represents a plan for processing and exporting features.

    This class encapsulates the proposed feature collection, column mappings,
    and metadata.
    """

    def __init__(
        self,
        type: str,
        planned_collection: FeatureCollection,
        column_mappings: dict[str, str],
        ignore_selectors: bool = False,
    ):
        """
        Initializes a feature plan.

        :param type: The type of the feature plan (e.g., "grid-daily-average").
        :type type: str
        :param planned_collection: The proposed feature collection.
        :type planned_collection: FeatureCollection
        :param column_mappings: A mapping of exported column names to desired column names.
        :type column_mappings: dict[str, str]
        :param ignore_selectors: Whether to ignore selectors during processing. Defaults to False.
        :type ignore_selectors: bool, optional
        """
        self.type = type
        self.planned_collection = planned_collection
        self.column_mappings = column_mappings
        self.ignore_selectors = ignore_selectors

    @property
    def intermediate_columns(self) -> list[str]:
        """
        Returns the columns that will be exported to the intermediate storage.

        :return: The keys of the column_mappings dictionary.
        :rtype: list[str]
        """
        return list(self.column_mappings.keys())

    @property
    def wanted_columns(self) -> list[str]:
        """
        Returns the columns that are wanted in the final export.

        :return: The values of the column_mappings dictionary.
        :rtype: list[str]
        """
        return list(self.column_mappings.values())
