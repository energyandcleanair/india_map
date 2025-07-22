"""Tests for the PipelineConfig class, ensuring its attributes and methods work as expected."""

from pm25ml.collectors.export_pipeline import PipelineConfig, AvailableIdKeys, ValueColumnType


def test__PipelineConfig__initialization__attributes_set_correctly():
    result_subpath = "test_path"
    id_columns: set[AvailableIdKeys] = {"date", "grid_id"}
    value_column_type_map = {
        "value1": ValueColumnType.FLOAT,
        "value2": ValueColumnType.INT,
    }
    expected_n_rows = 100

    config = PipelineConfig(result_subpath, id_columns, value_column_type_map, expected_n_rows)

    assert config.result_subpath == result_subpath
    assert config.id_columns == id_columns
    assert config.value_column_type_map == value_column_type_map
    assert config.expected_rows == expected_n_rows


def test__PipelineConfig__all_columns__returns_union_of_columns():
    id_columns: set[AvailableIdKeys] = {"date", "grid_id"}
    value_column_type_map = {
        "value1": ValueColumnType.FLOAT,
        "value2": ValueColumnType.INT,
    }
    config = PipelineConfig("test_path", id_columns, value_column_type_map, 100)

    all_columns = config.all_columns

    assert all_columns == id_columns.union(value_column_type_map.keys())
