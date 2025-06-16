import arrow
import pytest

from pm25ml.collectors.ned.coord_types import Lat, Lon


def test__NedDataRetriever__stream_files__not_implemented():
    """Test that the stream_files method raises NotImplementedError."""
    from pm25ml.collectors.ned.data_retrievers import NedDataRetriever
    from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor

    retriever = NedDataRetriever()
    descriptor = NedDatasetDescriptor(
        dataset_name="test_dataset",
        dataset_version="1.0",
        start_date=arrow.get("2023-01-01"),
        end_date=arrow.get("2023-01-02"),
        filter_bounds=(Lon(-10.0), Lat(-10.0), Lon(10.0), Lat(10.0)),
        source_variable_name="source_var",
        target_variable_name="target_var",
    )

    with pytest.raises(
        NotImplementedError, match="This method should be implemented by subclasses."
    ):
        list(
            retriever.stream_files(dataset_descriptor=descriptor)
        )  # Convert to list to force iteration
