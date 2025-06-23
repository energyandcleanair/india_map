# PM2.5 predictions in India

The aim of this project is to regularly produce raster PM2.5 predictions at a 10 km resolution in
India. It features a two-stage machine learning model for daily PM2.5 predictions.

Based on *[Improved daily PM2.5 estimates in India reveal inequalities in recent enhancement of air quality]*.
This paper created results from 2005-2023 can be [downloaded from Zenodo].

**Citation**: Kawano, Ayako, Makoto Kelp, Minghao Qiu, Kirat Singh, Eeshan Chaturvedi, Sunil Dahiya,
Inés Azevedo, and Marshall Burke. "Improved daily PM2. 5 estimates in India reveal inequalities in
recent enhancement of air quality." Science Advances 11, no. 4 (2025): eadq1071.

## Project layout

The `pm25ml` is where most of the code for this project can be found.

We have additional directories:
 - `experiments`: Experiments to inform implementation and to understand reference code.
 - `reference`: The files from the original forked project.

## Contributing

### Getting started

We use poetry to manage the project. To install the dependencies needed run:
```
poetry install --only main,dev
```

### Dependencies

We use different poetry groups to manage the dependencies: `main` (default), `dev`, `experiment`,
and `reference`. `experiment` and `reference` are used for the `experiments` and `reference` directories.
When adding dependencies, make sure you add them to the correct group.

> [!IMPORTANT]
> Do not add any dependencies that use GDAL to the project. We avoid GDAL to simplify the environment
> for running the code.

### Testing

Add unit tests for new classes and functions. When committing, make sure the tests pass.

We use pytest for the tests. The test files live alongside with original file with the suffixes:
`__test.py` for unit tests or `__it.py` for integration tests.

You can run the unit tests from the command line with:
```
poetry run pytest -m "not integration"
```

And you can run the integration tests from the command line with:
```
poetry run pytest -m "integration"
```

The integration tests expect you to be already set up and authenticated in your environment to use:
 - A Google Cloud project
 - NASA Earthaccess
 - Google Earth Engine 
 - A bucket for test assets (with an environment variable `IT_GEE_ASSET_BUCKET_NAME` for the name set)
 - An environment variable set for `IT_GEE_ASSET_ROOT`, which is where you want the test assets to go
   in GEE.

### Code standards

We use the "ALL" rules configuration provided by ruff, with an extended line-length of 100
characters.

To make sure your code meets our code standards, install the [`pre-commit`](https://pre-commit.com/)
configuration provided so that your code is checked before committing:
1. Install `pre-commit`, if you haven't already
2. [Install the git hook scripts](https://pre-commit.com/#3-install-the-git-hook-scripts)

## Implementation

This shows the overall process flow for the application.

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart TB
  collect_station_data["Collect station data"]
  collect_features["Collect features"]
  impute_satellite["Impute satellite"]
  train_pm25_model["Train PM2.5 model"]
  predict_pm25["Predict PM2.5"]

  collect_features --> train_pm25_model
  collect_features --> impute_satellite
  collect_features --> predict_pm25

  impute_satellite --> train_pm25_model
  impute_satellite --> predict_pm25

  collect_station_data --> train_pm25_model

  train_pm25_model --> predict_pm25

```

### Imputing data

This shows the data needed to impute the data.

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart TB
  direction TB

  to_impute["`
    **NASA Earth Data**
    MERRA AOT
    MERRA CO
    OMI NO2
  `"]
  grid_to_impute{{"Grid"}}
  gridded_to_impute["`
    **Gridded NASA Earth Data**
  `"]

  gee_feature_sets["`
    **GEE feature sets**
    TROPOMI CO
    TROPOMI NO2
    AOD*
    Meteorology
    Land cover type
    Elevation
  `"]

  generated_feature_sets["`
    **Generated feature sets**
    Monsoon flag
  `"]

  imputation{{Impute}}

  imputed_data["`
    **Imputed data**
    TROPOMI NO2
    TROPOMI CO
    AOD
  `"]

  to_impute --> grid_to_impute --> gridded_to_impute

  gee_feature_sets --> imputation
  generated_feature_sets --> imputation
  gridded_to_impute --> imputation


  imputation --> imputed_data

  classDef missing fill:#630014,color:#ffc7d2;

  class to_impute,grid_to_impute,feature_sets,grid_feature_sets missing

```

### Training the model

This shows the data needed to train the model.

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart TB
  direction TB

  nasa_earth_data["`
    **NASA Earth Data**
    MERRA AOT
    MERRA CO
    OMI NO2
  `"]
  grid_nasa_earth_data{{"Grid"}}
  gridded_nasa_earth_data["`
    **Gridded NASA Earth Data**
  `"]

  gee_feature_sets["`
    **GEE feature sets**
    Meteorology
    Land cover type
    Elevation
  `"]

  generated_feature_sets["`
    **Generated feature sets**
    Monsoon flag
  `"]

  imputed_data["`
    **Imputed data**
  `"]

  station_data["`
    **Station data**
  `"]

  station_data_cleaning{{"`
    **Station data cleaning**
  `"}}

  clean_station_data["`
    Clean station data
  `"]

  training{{Training}}

  model["`
    **Model**
  `"]

  nasa_earth_data --> grid_nasa_earth_data --> gridded_nasa_earth_data

  station_data --> station_data_cleaning --> clean_station_data

  gee_feature_sets --> training
  generated_feature_sets --> training
  imputed_data --> training
  gridded_nasa_earth_data --> training
  clean_station_data --> training

  training --> model

  classDef missing fill:#630014,color:#ffc7d2;
  classDef from_elsewhere fill:#004517,color:#bdfcd2;

  class nasa_earth_data,grid_nasa_earth_data,feature_sets,grid_feature_sets,station_data_cleaning missing

  class imputed_data from_elsewhere


```

[Improved daily PM2.5 estimates in India reveal inequalities in recent enhancement of air quality]: https://www.science.org/doi/10.1126/sciadv.adq1071
[downloaded from Zenodo]: https://zenodo.org/records/13694585
