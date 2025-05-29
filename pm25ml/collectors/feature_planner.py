from ee import ImageCollection, Reducer, FeatureCollection, Image
from arrow import Arrow

from pm25ml.collectors.constants import INDIA_CRS, SCALE_10KM

ISO8601_WITHOUT_TZ = "YYYY-MM-DDTHH:mm:ss"
ISO8601_DATE_ONLY = "YYYY-MM-DD"


class FeatureCollectionPlanner:

    def __init__(self, grid: FeatureCollection):
        self.grid = grid

    def plan_grid_daily_average(
        self,
        *,
        collection_name: str,
        selected_bands: list[str],
        dates: list[Arrow],
    ) -> "FeaturePlan":

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
                scale=SCALE_10KM,
            ).map(carry_date_through)

        processed_images: FeatureCollection = images.map(
            average_grid_value_for_date
        ).flatten()

        return FeaturePlan(
            type="grid-daily-average",
            planned_collection=processed_images,
            column_mappings=column_mappings,
        )


class FeaturePlan:
    def __init__(
        self,
        type: str,
        planned_collection: FeatureCollection,
        column_mappings: dict[str, str],
    ):
        self.type = type
        self.planned_collection = planned_collection
        self.column_mappings = column_mappings

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
