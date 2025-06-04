from ee import ImageCollection, Reducer, FeatureCollection, Image
from arrow import Arrow

from pm25ml.collectors.constants import INDIA_CRS, SCALE_10KM

ISO8601_WITHOUT_TZ = "YYYY-MM-DDTHH:mm:ss"
ISO8601_DATE_ONLY = "YYYY-MM-DD"

class GriddedFeatureCollectionPlanner:
    """
    A planner for creating gridded feature collections from Earth Engine data.

    Reference: how the "scale" arguments works in Earth Engine: https://developers.google.com/earth-engine/guides/scale
    """

    def __init__(self, grid: FeatureCollection):
        self.grid = grid

    def plan_daily_average(
        self,
        *,
        collection_name: str,
        selected_bands: list[str],
        dates: list[Arrow],
    ) -> "FeaturePlan":
        
        original_raster_scale = self._get_collection_scale(collection_name)

        ids = ["date", "grid_id"]
        transformed_band_names = (
            [f"{band}_mean" for band in selected_bands]
            if len(selected_bands) > 1
            else ["mean"]
        )
        exported_properties = ids + transformed_band_names
        wanted_properties = ids + selected_bands
        column_mappings = {
            exported: wanted
            for exported, wanted in zip(exported_properties, wanted_properties)
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
            carry_date_through = lambda f: f.set("date", image_date)
            return im.reduceRegions(
                collection=self.grid,
                # Single value per grid cell for the day.
                reducer=Reducer.mean(),
                crs=INDIA_CRS,
                scale=original_raster_scale,
            ).map(carry_date_through)

        processed_images: FeatureCollection = images.map(
            average_grid_value_for_date
        ).flatten()

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
        original_raster_scale = self._get_image_scale(image_name)

        ids = ["grid_id"]
        transformed_band_names = (
            [f"{band}_mean" for band in selected_bands]
            if len(selected_bands) > 1
            else ["mean"]
        )
        exported_properties = ids + transformed_band_names
        wanted_properties = ids + selected_bands
        column_mappings = {
            exported: wanted
            for exported, wanted in zip(exported_properties, wanted_properties)
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
            scale=original_raster_scale
        )

        flattened = collection_for_year.flatten()

        id_columns = ["grid_id"]
        wanted_columns = id_columns + list(
            output_names_to_class_values.keys()
        )
        exported_columns = id_columns + [
            f"{output_name}_mean" for output_name in output_names_to_class_values.keys()
        ]

        column_mappings = {
            exported: wanted
            for exported, wanted in zip(exported_columns, wanted_columns)
        }

        return FeaturePlan(
            type="annual-classified-pixels",
            planned_collection=flattened,
            column_mappings=column_mappings,
            ignore_selectors=True,
        )
    
    @staticmethod
    def _get_collection_scale(collection_name: str):
        return ImageCollection(collection_name).first().projection().nominalScale()

    @staticmethod
    def _get_image_scale(image_name: str):
        return Image(image_name).projection().nominalScale()

class FeaturePlan:
    def __init__(
        self,
        type: str,
        planned_collection: FeatureCollection,
        column_mappings: dict[str, str],
        ignore_selectors: bool = False,
    ):
        self.type = type
        self.planned_collection = planned_collection
        self.column_mappings = column_mappings
        self.ignore_selectors = ignore_selectors

    @property
    def intermediate_columns(self) -> list[str]:
        """
        Returns the columns that will be exported to the intermediate storage.
        These are the keys of the column_mappings dictionary.
        """
        return list(self.column_mappings.keys())

    @property
    def wanted_columns(self) -> list[str]:
        """
        Returns the columns that are wanted in the final export.
        These are the values of the column_mappings dictionary.
        """
        return list(self.column_mappings.values())
