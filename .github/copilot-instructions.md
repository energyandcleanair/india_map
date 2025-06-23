## Project overview

This is a Python project that is designed to predict PM2.5 levels in India using
two layers of machine learning models from satellite data using AQ station data.

### Core functionality

This project includes all of the code to:
 - Fetch the data from various satellite data sources and store it in parquet on
   Google Cloud Storage
 - Combine data into a wide format
 - Add additional computed features to the data
 - Train and run the machine learning models

## Copilot edits prime directive

 - Write well typed code that is easy to read and understand, and is secure.
 - Use the latest Python features and libraries.

## Preferred technologies

 - Polars for computed data
 - xarray for raw data

## Naming conventions

 - Test files are named: `<module_name>__test.py` or `<module_name>__it.py`
 - Test functions are named: `test__<thing_under_test>__<situation_under_test>__<expected_outcome>`
