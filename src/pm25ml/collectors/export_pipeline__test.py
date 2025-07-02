"""Tests for the PipelineConfig class, ensuring its attributes and methods work as expected."""

import pytest
from pm25ml.collectors.export_pipeline import PipelineConfig, AvailableIdKeys


def test__PipelineConfig__initialization__attributes_set_correctly():
    result_subpath = "test_path"
    id_columns: set[AvailableIdKeys] = {"date", "grid_id"}
    value_columns = {"value1", "value2"}
    expected_n_rows = 100

    config = PipelineConfig(result_subpath, id_columns, value_columns, expected_n_rows)

    assert config.result_subpath == result_subpath
    assert config.id_columns == id_columns
    assert config.value_columns == value_columns
    assert config.expected_rows == expected_n_rows


def test__PipelineConfig__all_columns__returns_union_of_columns():
    id_columns: set[AvailableIdKeys] = {"date", "grid_id"}
    value_columns = {"value1", "value2"}
    config = PipelineConfig("test_path", id_columns, value_columns, 100)

    all_columns = config.all_columns

    assert all_columns == id_columns.union(value_columns)
