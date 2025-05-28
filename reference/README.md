

## 1. Collect input features
- Scripts to download input features from Google Earth Engine have been included in the repository. 
- The rest of the features, namely MERRA-2 CO, MERRA-2 AOT, and OMI NO2, need to be collected separately by downloading the raw data from NASA Earthdata and values needed to be extracted to each grid in the grid_india_10km shapefiles.  

## 2. First stage ML to impute missing data in TROPOMI NO2, TROPOMI CO, and MODIS AOD
- LightGBM for NO2
- XGBoost for CO and AOD

## 3. Second stage ML to predict PM2.5 concentrations across India
