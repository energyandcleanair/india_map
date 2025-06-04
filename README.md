# PM2.5 predictions in India
- Two-stage machine learning model for daily PM2.5 predictions at a 10 km resolution in India, from 2005 to 2023
- Published in Science Advances. https://www.science.org/doi/10.1126/sciadv.adq1071
- Output PM2.5 estimates across India are publicly available and can be downloaded here: https://zenodo.org/records/13694585

## Citation
Kawano, Ayako, Makoto Kelp, Minghao Qiu, Kirat Singh, Eeshan Chaturvedi, Sunil Dahiya, Inés Azevedo, and Marshall Burke. "Improved daily PM2. 5 estimates in India reveal inequalities in recent enhancement of air quality." Science Advances 11, no. 4 (2025): eadq1071.

### Contributing

#### Code standards

We use the "ALL" rules configuration provided by ruff, with an extended line-length of 100
characters.

To make sure your code meets our code standards, install the [`pre-commit`](https://pre-commit.com/)
configuration provided so that your code is checked before committing:
1. Install `pre-commit`, if you haven't already
2. [Install the git hook scripts](https://pre-commit.com/#3-install-the-git-hook-scripts)


### Process dependencies overview

This shows the overall process flow and dependencies for the modelling.

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
