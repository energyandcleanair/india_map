# PM2.5 predictions in India

The aim of this project is to regularly produce raster PM2.5 predictions at a 10 km resolution in
India. It features a two-stage machine learning model for daily PM2.5 predictions.

Based on *[Improved daily PM2.5 estimates in India reveal inequalities in recent enhancement of air quality]*.
This paper created results from 2005-2023 can be [downloaded from Zenodo].

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

> [!IMPORTANT]
> When adding new integration tests, these must be marked with `pytest.mark.integration`.

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

This shows the overall process flow for the application. More detail is provided on some of these
areas in the document below.

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart TB
  collect_station_data["Collect station data"]
  collect_features["Collect and prepare features"]
  impute_satellite["Impute satellite using ML"]
  train_pm25_model["Train PM2.5 model"]
  predict_pm25["Predict PM2.5"]
  recombine_datasets["Combine collected, imputed, and station data"]

  collect_features --> impute_satellite

  impute_satellite --> recombine_datasets
  
  recombine_datasets --> train_pm25_model
  recombine_datasets --> predict_pm25

  collect_features --> recombine_datasets
  collect_station_data --> recombine_datasets

  train_pm25_model --> predict_pm25

```

### Collect and prepare features

We collect and combine data for each month in the analysis period.

Simple imputation that happens at this stage can include spatial or temporal imputation using
nearest neighbour or interpolation.

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart TB
  collect_gee_feature_sets{{"`
    **Collect GEE feature sets**
    *Regrids and processes data*
    TROPOMI CO
    TROPOMI NO2
    AOD*
    ERA5 land data
    ERA5 land cover type
    Elevation
  `"}}

  gee_feature_sets["**GEE feature sets**"]

  collect_ned{{"`
    **Collect NASA Earth Data**
    MERRA AOT
    MERRA CO
    OMI NO2
  `"}}
  ned_regrid{{"Regrid"}}
  ned_regridded["`
    **Gridded NASA Earth Data**
  `"]

  static_datasets["`
    **Static datasets**
    India 10km grid
    India 50km grid
    Region clustering
  `"]

  combine_datasets{{"**Combine datasets**"}}

  combined_datasets["**Combined datsets**"]

  simple_imputation{{"`
    **Impute minor missing data**
    Spatial ERA5 land coastal
  `"}}
  simple_imputed_datasets["Combined w/ simple imputation"]

  generate_features{{"`
    **Generate features sets**
    Monsoon flag
    Date related features
    Moving averages
    Combined features
  `"}}

  prepared_features["Prepared features"]

  collect_gee_feature_sets --> gee_feature_sets
  collect_ned --> ned_regrid --> ned_regridded

  gee_feature_sets --> combine_datasets
  ned_regridded --> combine_datasets
  static_datasets --> combine_datasets

  combine_datasets --> combined_datasets

  combined_datasets --> simple_imputation
  combined_datasets -->|extended| simple_imputed_datasets
  simple_imputation --> simple_imputed_datasets

  simple_imputed_datasets --> generate_features
  simple_imputed_datasets -->|extended| prepared_features

  generate_features --> prepared_features

  

```

### Impute satellite using ML

This shows the processing that happens at the satellite imputation stage.

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart TB

  prepared_features["`
    **Prepared features**
    From _Collect and prepare features_
  `"]

  subgraph for_each["`
  **For each:** AOD, NO2, CO
  `"]
    subgraph imputation["ML imputation"]
      sample{{"Sample prepared"}}
      train{{"Train model"}}
      impute{{"Impute missing"}}

      sample --> train
      train --> impute
    end
  end

  combined["Combined with imputed"]

  prepared_features --> sample  
  prepared_features --> impute
  prepared_features -->|extended| combined
  impute --> combined

```

### Train PM2.5 model

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

  classDef from_elsewhere fill:#004517,color:#bdfcd2;

  class imputed_data from_elsewhere


```

## Citations

### Models

Kawano, Ayako, Makoto Kelp, Minghao Qiu, Kirat Singh, Eeshan Chaturvedi, Sunil Dahiya,
Inés Azevedo, and Marshall Burke. "Improved daily PM2. 5 estimates in India reveal inequalities in
recent enhancement of air quality." Science Advances 11, no. 4 (2025): eadq1071.

### Bundled test datasets

#### `M2T1NXAER.5.12.4_MERRA2_400.tavg1_2d_aer_Nx.20230101_TOTEXTTAU_subsetted.nc4`

NASA Global Modeling and Assimilation Office (GMAO). (2015).
*MERRA-2 tavg1_2d_aer_Nx: Aerosol Diagnostics, Hourly 0.5° × 0.625°, V5.12.4 (M2T1NXAER)* [Data set].
Goddard Earth Sciences Data and Information Services Center (GES DISC), NASA GSFC.
https://doi.org/10.5067/KLICLTZ8EM9D  
(Accessed 24 Jun 2025; licence — CC-0 1.0. NASA does not endorse this software.)

#### `OMI-Aura_L3-OMNO2d_2023m0111_v003-2023m0223t191034.he5`

NASA Goddard Space Flight Center. (2023).
*OMI/Aura NO₂ Cloud-Screened Total and Tropospheric Column L3 Global Gridded 0.25° × 0.25° V003* (OMNO2d) [Data set].
NASA Goddard Earth Sciences Data and Information Services Center (GES DISC).
https://doi.org/10.5067/Aura/OMI/DATA3002  
(Accessed 24 Jun 2025; licence — CC-0 1.0. NASA does not endorse this software.)

[Improved daily PM2.5 estimates in India reveal inequalities in recent enhancement of air quality]: https://www.science.org/doi/10.1126/sciadv.adq1071
[downloaded from Zenodo]: https://zenodo.org/records/13694585
