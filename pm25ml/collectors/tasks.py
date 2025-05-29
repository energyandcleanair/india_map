from time import sleep
import uuid
from ee import ImageCollection, Reducer, FeatureCollection, Image
from ee.batch import Export, Task
from arrow import Arrow, now

from pm25ml.collectors.constants import INDIA_CRS, SCALE_10KM

ISO8601_WITHOUT_TZ = "YYYY-MM-DDTHH:mm:ss"
ISO8601_DATE_ONLY = "YYYY-MM-DD"


class TaskBuilder:

    def __init__(self, grid: FeatureCollection, bucket_name: str):
        self.grid = grid
        self.bucket_name = bucket_name

    def build_grid_daily_average_task(
        self,
        *,
        collection_name: str,
        selected_bands: list[str],
        dates: list[Arrow],
    ) -> "GriddingTask":

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

        processed_images = (
            images.map(average_grid_value_for_date)
            .flatten()
        )

        # We create a unique file name prefix for the export.
        file_name_prefix = str(uuid.uuid4())

        cleaned_collection_name = (
            collection_name.replace("/", "-")
            .replace(":", "-")
            .replace(".", "-")
            .replace("_", "-")
            .replace(" ", "-")
            .lower()
        )
        task_name = f"grid-daily-average_{cleaned_collection_name}_{now().format('YYYY-MM-DD_HH-mm-ss')}"

        task = Export.table.toCloudStorage(
            description=task_name,
            collection=processed_images,
            bucket=self.bucket_name,
            fileNamePrefix=file_name_prefix,
            fileFormat="CSV",
            selectors=exported_properties,
        )

        return GriddingTask(
            task=task,
            description=task_name,
            file_name_prefix=file_name_prefix,
            bucket_name=self.bucket_name,
            column_mappings=column_mappings,
        )

class GriddingTaskError(Exception):
    pass


class GriddingTask:
    def __init__(
        self,
        *,
        description: str,
        task: Task,
        file_name_prefix: str,
        column_mappings: dict[str, str],
        bucket_name: str,
    ):
        self.description = description
        self.task = task
        self.file_name_prefix = file_name_prefix
        self.column_mappings = column_mappings
        self.bucket_name = bucket_name

    def complete_task(self):
        self.task.start()
        delay_backoff = 1
        growth_factor = 1.5
        max_delay = 60
        while self.task.active():
            sleep(delay_backoff)
            delay_backoff = min(max_delay, delay_backoff * growth_factor)

        if not self.success:
            raise GriddingTaskError(
                f"Task failed ({self.task.status()}): {self.error_message}"
            )

        return self

    @property
    def status(self):
        return self.task.status().get("state", None)

    @property
    def success(self):
        return self.status == "COMPLETED"

    @property
    def error_message(self):
        if not self.success:
            return self.task.status().get("error_message", "No error message provided.")
        return None

    @property
    def result_bucket_path(self):
        return f"{self.bucket_name}/{self.file_name_prefix}.csv"
