import math
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
from xgboost import XGBRegressor


def main():
    '''Run imputation ML model for AOD'''

    # 0. Read data
    df = get_data_for_imputation(
        input_file="../../data_from_ayako/df_for_imputation_test.parquet",
        supergrid_file="../../data_from_ayako/grid_intersect_with_50km.csv",
        training=True)

    # 1. Sampling
    df_sampled, df_rest = sample_data(df)

    # # for faster testing, take every 100th row
    # df_sampled = df_sampled.iloc[::100, :].reset_index(drop=True)

    # 2. Create folds
    # outer_cv is a list where each item contains a tuple with indices
    # of training and validation sets for each fold
    outer_cv = make_folds(df_sampled, n_folds=10)

    # 3. Finding optimal hyperparameters (inner CV)
    # - if GPU's available, can use tree_method='gpu_hist'
    best_params_XGB, hyper_tuning_metrics = find_hyper_params(
        df_sampled, outer_cv[0], tree_method='hist')

    # best_params_XGB = {'subsample': 0.8, 'n_estimators': 1000, 'min_child_weight': 1,
    #                    'max_depth': 20, 'lambda': 100, 'gamma': 0.8, 'eta': 0.1, 'booster': 'gbtree'}

    # 4. Training imputation model (fit XGBRegressor, compute training metrics)
    model, training_diagnostics = train_model(
        df_sampled, outer_cv, best_params_XGB, tree_method='hist')

    # 5. Evaluate model on the test data
    test_metrics = evaluate_model(model, df_rest)

    # 6. Impute missing values using the trained model
    # TODO update to read df for each month separately
    df_to_impute = get_data_for_imputation(
        input_file="../../data_from_ayako/df_for_imputation_test.parquet",
        supergrid_file="../../data_from_ayako/grid_intersect_with_50km.csv",
        training=False)

    # Impute missing values in the 'aod' column
    # Note: this will not change the original df_to_impute, but return a new dataframe
    df_imputed = df_to_impute.copy()
    df_imputed['aod'] = model.predict(
        df_imputed.drop(columns=['aod', 'date', 'grid_id', 'grid_id_50km', 'year_month']))

    # TODO save the imputed dataframe to a file

    print("Imputation completed successfully.")


def get_data_for_imputation(input_file, supergrid_file, training=True):
    """Read data for imputation modelling, and add missing columns

    Following the code from AK, reading the dataframe for imputation
    modelling, only keep the columns needed, and add missing columns
    (50km grid id, year_month).

    In the future, this not needed as df comes ready from the file.

    Args:
        input_file (str): Path to the input file with data for imputation model.
        supergrid_file (str): Path to the file with 50km grid data.
        training (bool): If True, the function will return the dataframe ready for
            training, i.e. with no nans in the data. If False, the function will
            return the dataframe with nans in the column to be imputed..

    """

    # ~ ~ ~ Original code ~ ~ ~
    # df = pd.read_csv(
    #     "/oak/stanford/groups/mburke/pm_prediction/data/intermediate/ML_full_model/AOD_impute/aod_ml_df.csv")
    # print(df.shape)
    # print(df.columns)
    # pd.set_option('display.max_rows', None)
    # print(df.isna().sum())
    # df = df.drop(columns=['NO2_tropos', 'NO2_missing',
    #                     'aod_missing', 'CO', 'CO_missing',  'aod_intped', 'omi_no2', 'pressure_allyears',
    #                     'dewpoint_temp_allyears', 'pressure_annual', 'temp_annual',
    #                     'temp_allyears', 'low_veg_allyears', 'high_veg_allyears', 'RH_allyears', 'wind_degree_allyears',
    #                     'thermal_radiation_allyears', 'rainfall_allyears', 'u_wind_allyears', 'v_wind_allyears',
    #                     'omi_no2_allyears', 'aot_daily_allyears', 'pressure_rolling'])
    # grid_50km = pd.read_csv(
    #     "/scratch/users/akawano/pm_prediction/intermediate/grid_intersect_with_50km.csv")
    # print(grid_50km.shape)
    # df['grid_id'] = df['grid_id'].astype(int).astype(str)
    # grid_50km = grid_50km.rename(columns={'grid_id_10km': 'grid_id'})
    # grid_50km['grid_id'] = grid_50km['grid_id'].astype(int).astype(str)
    # df = pd.merge(df, grid_50km, how='left', on='grid_id')
    # print(df.shape)
    # df['year_month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    # ~ ~ ~ Original code end ~ ~ ~

    # 1. Read data for imputation model and keep needed columns
    df = pd.read_parquet(input_file)

    # NOTE: not checked against paper, the column list was created based on the
    # list of columns used in the prediction step
    # TODO check against the list of features in the paper
    aod_imput_cols = ['grid_id', 'date', 'aot_daily', 'co_daily', 'v_wind', 'u_wind',
                      'rainfall', 'temp', 'pressure', 'thermal_radiation', 'low_veg',
                      'high_veg', 'dewpoint_temp', 'aod', 'elevation', 'water', 'shurub',
                      'urban', 'forest', 'savannas', 'month', 'day_of_year',
                      'cos_day_of_year', 'monsoon', 'lon', 'lat', 'wind_degree', 'RH',
                      'aot_rolling', 'co_rolling', 'omi_no2_rolling', 'v_wind_rolling',
                      'u_wind_rolling', 'rainfall_rolling', 'temp_rolling',
                      'wind_degree_rolling', 'RH_rolling', 'thermal_radiation_rolling',
                      'dewpoint_temp_rolling', 'aot_daily_annual', 'co_daily_annual',
                      'omi_no2_annual', 'v_wind_annual', 'u_wind_annual', 'rainfall_annual',
                      'thermal_radiation_annual', 'low_veg_annual', 'high_veg_annual',
                      'dewpoint_temp_annual', 'wind_degree_annual', 'RH_annual',
                      'co_daily_allyears']

    df = df[aod_imput_cols]

    # For training the ML model, no nans allowed in the data. For imputing, only
    # the rows where data is missing is used.

    if training:
        # if training, drop rows with nans in the column to be imputed
        df = df.dropna(subset=['aod'])

        # In original code, the sum of nans is printed out. Since ML models don't like nans,
        # and in the data given there are no nans, assuming that the idea here is to check
        # that there are no nans in the data except in the column to be imputed
        if not (df.isna().sum() == 0).all():
            raise ValueError("No nans allowed in data for imputation")

    else:
        # keep all rows where aod is nan
        df = df[df['aod'].isna()]

        # still no nans allowed in the other columns in the data
        if not (df.drop(columns=['aod']).isna().sum() == 0).all():
            raise ValueError(
                "No nans allowed in data for imputation except in the column to be imputed")

    # 2. Add 50 km grid to dataframe
    df_shape = df.shape

    grid_50km = pd.read_csv(supergrid_file)
    grid_50km = grid_50km.rename(columns={'grid_id_10km': 'grid_id'})

    # Note: I'm not sure this is needed, as long as both dataframes have the grid_id in the same type
    df['grid_id'] = df['grid_id'].astype(int).astype(str)
    grid_50km['grid_id'] = grid_50km['grid_id'].astype(int).astype(str)

    df = pd.merge(df, grid_50km, how='left', on='grid_id')

    # check that after merging the shape of df has only changed by
    # number of columns increasing, and no change in the number of rows
    if not ((df_shape[0] == df.shape[0]) and (df_shape[1]+1 == df.shape[1])):
        raise ValueError("Something went wrong with merging 50km grid to data")

    # 3. Add column with year-month string
    # the parquet file already includes a column year_month, but the string
    # is in a different format, so overwriting with the string as done in original code
    # NOTE that this probably is not needed, and if it, could skip
    # reading this column from the file
    df['year_month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')

    return df


def sample_data(df):
    """Split data to training and test sets

    Following method from AK to use pandas method to randomly sample
    a fixed fraction of the data set for training. The rest is used
    for evaluation of the model.
    """

    # ~ ~ ~ Original code ~ ~ ~
    # # randomly sample the dataframe based on grid_id_50km, month, and year
    # df_sampled = df.groupby(['grid_id_50km', 'year_month']).apply(lambda x: x.sample(
    #     frac=0.03, random_state=42, replace=False)).reset_index(drop=True)
    # print("randomly sampled dataframe")
    # print(df_sampled.shape)
    # print("number of unique grid_id in the sampled dataset")
    # print(df_sampled['grid_id'].nunique())
    # print("number of unique grid_id 50km in the sampled dataset")
    # print(df_sampled['grid_id_50km'].nunique())
    # df_sampled.to_csv(
    #     "/oak/stanford/groups/mburke/pm_prediction/data/intermediate/ML_full_model/AOD_impute/aod_ml_df_sampled.csv", index=False)
    # # remained data
    # df_sampled['grid_date'] = df_sampled['grid_id'].astype(
    #     str) + "_" + df_sampled['date'].astype(str)
    # df['grid_date'] = df['grid_id'].astype(str) + "_" + df['date'].astype(str)
    # rest_df = df[~df['grid_date'].isin(df_sampled['grid_date'])]
    # df_sampled = df_sampled.drop(columns='grid_date')
    # df = df.drop(columns='grid_date')
    # ~ ~ ~ Original code end ~ ~ ~

    # Add grid_date column, used to identify rows not in the sampled dataset
    df['grid_date'] = df['grid_id'].astype(str) + "_" + df['date'].astype(str)

    # sample the training set
    # https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.sample.html
    # df_sampled will still contain nan values for aod
    df_sampled = df.groupby(['grid_id_50km', 'year_month']).apply(lambda x: x.sample(
        frac=0.03, random_state=42, replace=False)).reset_index(drop=True)

    # logging for training set
    print(f"Shape of dataframe used for impuation modelling: {df.shape}")
    print(
        f"Shape of randomly sampled dataframe for training: {df_sampled.shape}")
    print(
        f"Number of unique grid_id in the sampled dataset: {df_sampled['grid_id'].nunique()}")
    print(
        f"Number of unique grid_id 50km in the sampled dataset: {df_sampled['grid_id_50km'].nunique()}")

    # get the rows not in the training set, these used for validation
    df_rest = df[~df['grid_date'].isin(df_sampled['grid_date'])]

    # check: number of rows in df_sample and df_rest should equal the
    # number of rows in df
    if df_sampled.shape[0]+df_rest.shape[0] != df.shape[0]:
        raise Exception("Something went wrong in the sampling")

    # drop the grid_date columns, it was only used for generating df_rest
    df = df.drop(columns='grid_date')

    return df_sampled.drop(columns='grid_date'), df_rest.drop(columns='grid_date')


def make_folds(df_sampled, n_folds: int):
    """Make the folds for inner and outer cross-validation

    Cross-validation is used to ensure that the model is trained and evaluated
    on different data, for intro see https://scikit-learn.org/stable/modules/cross_validation.html
    Here, df_sampled should already be the training data set, with the test set
    being set appart before. This function creates the cross-validation folds
    and returns the indexes for the training and validation sets for each fold.

    Using GroupKFold to ensure that the same grid_id_50km is not in both
    training and testing sets.

    https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold.split

    """

    # ~ ~ ~ Original code ~ ~ ~
    # df_sampled_copy = df_sampled.copy()
    # y = df_sampled_copy.pop('aod').to_frame()
    # X = df_sampled_copy
    # gkf = GroupKFold(n_splits=10)
    # outer_cv = gkf.split(X, y, groups=X['grid_id_50km'])
    # # Convert the generator to a list
    # outer_cv = list(outer_cv)
    # ~ ~ ~ Original code end ~ ~ ~

    y = df_sampled[['aod']]
    X = df_sampled.drop(columns=['aod'])

    # NOTE because using the default shuffle=False, not possible to give the random_state
    # is the result reproducible?
    gkf = GroupKFold(n_splits=n_folds)

    # split returns the indices of the training and testing sets
    outer_cv = gkf.split(X, y, groups=X['grid_id_50km'])
    outer_cv = list(outer_cv)

    return outer_cv


def find_hyper_params(df_sampled, selected_outer_cv, tree_method):
    """Find the optimal hyperparameters for the model

    This function performs hyperparameter tuning using RandomizedSearchCV
    on the XGBRegressor model. It uses data from 1 one of the outer 
    cross validation folds. 

    The paper states that for AOD imputation, the XGBoost model was used.
    Therefore ingoring the part with LightGBM in the original code.

    Args:
        df_sampled (pd.DataFrame): Dataframe with sampled data for training.
        selected_outer_cv (tuple): A tuple containing the indices of the
            training and validation sets for the selected outer fold.
        tree_method (str): The tree method to use for XGBRegressor.
    """

    # # ~ ~ ~ Original code ~ ~ ~
    # inner_train_indices, inner_test_indices = outer_cv[0]
    # inner_train = df_sampled.loc[inner_train_indices]
    # inner_train = shuffle(inner_train).reset_index(drop=True)

    # y_inner = inner_train.pop('aod').to_frame()
    # X_inner = inner_train
    # gkf = GroupKFold(n_splits=5)
    # inner_cv = gkf.split(X_inner, y_inner, groups=X_inner['grid_id_50km'])
    # inner_cv = list(inner_cv)
    # X_inner = X_inner.drop(
    #     columns=['grid_id', 'date', 'year', 'grid_id_50km', 'year_month'])
    # # # # # # # # # # LightGBM # # # # # # # # # # # # #
    # pipe_LGBM = LGBMRegressor()
    # params_LGBM = {'max_depth': [10],
    #             'learning_rate': [0.1],
    #             'num_iterations': [3000],  # 3000
    #             # 'early_stopping_rounds':[300],
    #             'num_leaves': [1500],  # , 1500
    #             'max_bin': [2000],  # , 350
    #             'min_data_in_leaf': [10],  # , 40
    #             'lambda_l2': [1],  # , 500
    #             'boosting': ['gbdt'],  # default: gbdt, option: 'dart',
    #             'objective': ['regression']
    #             }
    # # {'boosting': 'gbdt', 'lambda_l2': 100, 'learning_rate': 0.1, 'max_bin': 3000, 'max_depth': 10, 'min_data_in_leaf': 10, 'num_iterations': #3000, 'num_leaves': 1500, 'objective': 'regression'}
    # # LGBM Results
    # # ===================================
    # # test r2
    # # 0.8046384256718779
    # # train r2
    # # 0.9602720829361677
    # # test rmse
    # # -160.15591178085523
    # # train rmse
    # # -72.2707241547296
    # scoring = {'r_squared': 'r2', 'rmse': 'neg_root_mean_squared_error'}
    # LGBM_search = GridSearchCV(pipe_LGBM, params_LGBM, n_jobs=int(os.getenv("SLURM_CPUS_PER_TASK")),
    #                         scoring=scoring, refit="rmse", cv=inner_cv, verbose=1, return_train_score=True)
    # LGBM_search.fit(X_inner, y_inner.values.ravel())
    # LGBM_results = LGBM_search.cv_results_
    # best_params_LGBM = LGBM_search.best_params_
    # print("LGBM best parameters")
    # print(best_params_LGBM)
    # for i, x in enumerate(LGBM_results['params']):
    #     if x == best_params_LGBM:
    #         index = i
    #         break
    # else:
    #     print("best_params is not found in a")
    # val_r2_list = [LGBM_results['split0_test_r_squared'][index], LGBM_results['split1_test_r_squared'][index],
    #             LGBM_results['split2_test_r_squared'][index], LGBM_results['split3_test_r_squared'][index],
    #             LGBM_results['split4_test_r_squared'][index]]
    # train_r2_list = [LGBM_results['split0_train_r_squared'][index], LGBM_results['split1_train_r_squared'][index],
    #                 LGBM_results['split2_train_r_squared'][index], LGBM_results['split3_train_r_squared'][index],
    #                 LGBM_results['split4_train_r_squared'][index]]
    # val_rmse_list = [LGBM_results['split0_test_rmse'][index], LGBM_results['split1_test_rmse'][index],
    #                 LGBM_results['split2_test_rmse'][index], LGBM_results['split3_test_rmse'][index],
    #                 LGBM_results['split4_test_rmse'][index]]
    # train_rmse_list = [LGBM_results['split0_train_rmse'][index], LGBM_results['split1_train_rmse'][index],
    #                 LGBM_results['split2_train_rmse'][index], LGBM_results['split3_train_rmse'][index],
    #                 LGBM_results['split4_train_rmse'][index]]
    # val_r2 = LGBM_results['mean_test_r_squared'][index]
    # train_r2 = LGBM_results['mean_train_r_squared'][index]
    # val_rmse = LGBM_results['mean_test_rmse'][index]
    # train_rmse = LGBM_results['mean_train_rmse'][index]
    # print("LGBM Results")
    # print("===================================")
    # print("test r2")
    # print(val_r2)
    # print("train r2")
    # print(train_r2)
    # print("test rmse")
    # print(val_rmse)
    # print('train rmse')
    # print(train_rmse)
    # # # # # # # # # # XGBoost # # # # # # # # # # # # #
    # pipe_XGB = XGBRegressor(tree_method='gpu_hist')
    # params_XGB = {'eta': [0.1],
    #             'gamma': [0.8],
    #             'max_depth': [20],
    #             'min_child_weight': [1],
    #             'subsample': [0.8],
    #             'lambda': [100],
    #             'n_estimators': [1000],
    #             'booster': ['gbtree']
    #             }
    # # {'subsample': 0.8, 'n_estimators': 1000, 'min_child_weight': 1, 'max_depth': 20, 'lambda': 100, 'gamma': 0.8, 'eta': 0.1, 'booster': #'gbtree'}
    # # XGB Results
    # # ===================================
    # # test r2
    # # 0.812578428261584
    # # train r2
    # # 0.9993900207043313
    # # test rmse
    # # -156.86442208566336
    # # train rmse
    # # -8.949776397157246
    # scoring = {'r_squared': 'r2', 'rmse': 'neg_root_mean_squared_error'}
    # XGB_search = RandomizedSearchCV(pipe_XGB, params_XGB, n_jobs=int(os.getenv("SLURM_CPUS_PER_TASK")),
    #                                 scoring=scoring, refit="rmse", cv=inner_cv, verbose=1, return_train_score=True)
    # XGB_search.fit(X_inner, y_inner.values.ravel())
    # results = XGB_search.cv_results_
    # best_params_XGB = XGB_search.best_params_
    # print("XGB best parameters")
    # print(best_params_XGB)
    # for i, x in enumerate(results['params']):
    #     if x == best_params_XGB:
    #         index = i
    #         break
    # else:
    #     print("best_params is not found in a")
    # val_r2_list = [results['split0_test_r_squared'][index], results['split1_test_r_squared'][index],
    #             results['split2_test_r_squared'][index], results['split3_test_r_squared'][index],
    #             results['split4_test_r_squared'][index]]
    # train_r2_list = [results['split0_train_r_squared'][index], results['split1_train_r_squared'][index],
    #                 results['split2_train_r_squared'][index], results['split3_train_r_squared'][index],
    #                 results['split4_train_r_squared'][index]]
    # val_rmse_list = [results['split0_test_rmse'][index], results['split1_test_rmse'][index],
    #                 results['split2_test_rmse'][index], results['split3_test_rmse'][index],
    #                 results['split4_test_rmse'][index]]
    # train_rmse_list = [results['split0_train_rmse'][index], results['split1_train_rmse'][index],
    #                 results['split2_train_rmse'][index], results['split3_train_rmse'][index],
    #                 results['split4_train_rmse'][index]]
    # val_r2 = results['mean_test_r_squared'][index]
    # train_r2 = results['mean_train_r_squared'][index]
    # val_rmse = results['mean_test_rmse'][index]
    # train_rmse = results['mean_train_rmse'][index]
    # print("XGB Results")
    # print("===================================")
    # print("test r2")
    # print(val_r2)
    # print("train r2")
    # print(train_r2)
    # print("test rmse")
    # print(val_rmse)
    # print('train rmse')
    # print(train_rmse)
    # ~ ~ ~ Original code end ~ ~ ~

    # Get the training set for the selected outer fold
    # selected_outer_cv is a tuple of arrays, where the first array contains
    # the indices for the training set and the second array contains the indices
    # for the validation set for the selected outer fold
    inner_train = df_sampled.loc[selected_outer_cv[0]]

    # Schuffle the training set to ensure randomness. Could use a random_state
    # for reproducibility, but this not done in the original code.
    inner_train = shuffle(inner_train).reset_index(drop=True)

    # Separate the target variable 'aod' from the features
    y_inner = inner_train[['aod']]
    X_inner = inner_train.drop(columns=['aod'])

    # Create the inner cross-validation folds using GroupKFold
    gkf = GroupKFold(n_splits=5)
    inner_cv = gkf.split(X_inner, y_inner, groups=X_inner['grid_id_50km'])

    # Use RandomizedSearchCV to tune hyperparameters for XGBRegressor
    pipe_XGB = XGBRegressor(tree_method=tree_method)

    params_XGB = {'eta': [0.1],
                  'gamma': [0.8],
                  'max_depth': [20],
                  'min_child_weight': [1],
                  'subsample': [0.8],
                  'lambda': [100],
                  'n_estimators': [1000],
                  'booster': ['gbtree']
                  }

    # n_jobs could also be defined from environment variable, e.g. SLURM_CPUS_PER_TASK
    XGB_search = RandomizedSearchCV(pipe_XGB,
                                    params_XGB,
                                    n_jobs=1,
                                    scoring={
                                        'r_squared': 'r2', 'rmse': 'neg_root_mean_squared_error'},
                                    refit="rmse",
                                    cv=list(inner_cv),
                                    verbose=1,
                                    return_train_score=True)

    XGB_search.fit(X_inner.drop(columns=['grid_id', 'date', 'grid_id_50km', 'year_month']),
                   y_inner.values.ravel())

    # Get the best hyperparameters
    best_params_XGB = XGB_search.best_params_
    print("XGB best parameters")
    print(best_params_XGB)

    # Get the performance metrics for the best hyperparameters
    # TODO I'm getting user warning:
    # UserWarning: The total space of parameters 1 is smaller than n_iter=10. Running 1 iterations.
    # -> cannot check how the rest behaves when more than 1 result
    results = XGB_search.cv_results_

    # find the index of the best parameters in the results
    # TODO is there a better way to do this?
    for i, x in enumerate(results['params']):
        if x == best_params_XGB:
            index = i
            break
    else:
        print("best_params is not found in a")

    val_r2 = results['mean_test_r_squared'][index]
    train_r2 = results['mean_train_r_squared'][index]
    val_rmse = results['mean_test_rmse'][index]
    train_rmse = results['mean_train_rmse'][index]

    # Return the best hyperparameters and performance metrics
    return best_params_XGB, {
        "val_r2": val_r2,
        "train_r2": train_r2,
        "val_rmse": val_rmse,
        "train_rmse": train_rmse
    }


def train_model(df_sampled, outer_cv, best_params_XGB, tree_method='gpu_hist'):
    """Train the imputation model using XGBRegressor

    This function trains the XGBRegressor model using the sampled data
    and the outer cross-validation folds. It computes the training and
    validation scores and extracts feature importances for each fold.
    Unlike in the original code, the full predicted test and validation
    data are not stored (variable train_dfs and eval_dfs in original code).

    Args:
        df_sampled (pd.DataFrame): Dataframe with sampled data for training.
        outer_cv (list): List of tuples with indices for training and testing sets.
        best_params_XGB (dict): Dictionary with hyperparameters for XGBRegressor.
        tree_method (str): Method for tree construction in XGBRegressor, default 
            is 'gpu_hist' (following the original code). If no GPU available, use 'hist'.
    """

    # # ~ ~ ~ Original code ~ ~ ~
    # df_sampled_copy = df_sampled.copy()
    # y = df_sampled_copy.pop('aod').to_frame()
    # X = df_sampled_copy
    # ...
    # making folds & inner CV
    # ...
    #
    # # # # # # # # # # Outer CV using best parameters # # # # # # # # # # # # #
    # trn_r2 = []
    # trn_rmse = []
    # cv_r2 = []
    # cv_rmse = []
    # dfs = []
    # train_dfs = []
    # eval_dfs = []
    # for n_fold, (trn_idx, val_idx) in enumerate(outer_cv):
    #     print(f"========= fold:{n_fold} =========")
    #     # train_indices = [item for sublist in trn_idx for item in sublist]
    #     # test_indices = [item for sublist in val_idx for item in sublist]
    #     X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    #     y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    #     train_df = pd.DataFrame({'date': X_trn['date'], 'grid_id': X_trn['grid_id'],
    #                             'y_trn': y_trn['aod']})

    #     date = X_val['date']
    #     grid = X_val['grid_id']

    #     eval_df = pd.DataFrame({'date': date,
    #                             'grid_id': grid,
    #                             'y_val': y_val['aod']})

    #     X_trn = X_trn.drop(
    #         columns=['date', 'grid_id', 'grid_id_50km', 'year_month'])
    #     X_val = X_val.drop(
    #         columns=['date', 'grid_id', 'grid_id_50km', 'year_month'])

    #     best_params_XGB = {'subsample': 0.8, 'n_estimators': 1000, 'min_child_weight': 1,
    #                     'max_depth': 20, 'lambda': 100, 'gamma': 0.8, 'eta': 0.1, 'booster': 'gbtree'}

    #     model_xgb = XGBRegressor(**best_params_XGB, n_jobs=int(os.getenv(
    #         "SLURM_CPUS_PER_TASK")), tree_method='gpu_hist')  # , tree_method='gpu_hist'
    #     # model_lgbm = LGBMRegressor(**best_params_LGBM, n_jobs = int(os.getenv("SLURM_CPUS_PER_TASK")))
    #     model_xgb.fit(X_trn, y_trn.values.ravel())
    #     # model_lgbm.fit(X_trn, y_trn.values.ravel())

    #     importances = model_xgb.feature_importances_
    #     feature = X_trn.columns
    #     df = pd.DataFrame({'feature': feature, 'importance': importances}).sort_values(
    #         by=['importance'], ascending=False)
    #     dfs.append(df)

    #     trn_y_pred = model_xgb.predict(X_trn)
    #     trn_score = r2_score(y_trn, trn_y_pred)
    #     print(f"Training R2: {trn_score}")
    #     trn_r2.append(trn_score)
    #     train_df['trn_y_pred'] = trn_y_pred
    #     train_dfs.append(train_df)

    #     trn_MSE = mean_squared_error(y_trn, trn_y_pred)
    #     trn_RMSE = math.sqrt(trn_MSE)
    #     trn_rmse.append(trn_RMSE)
    #     print(f"Training RMSE: {trn_RMSE}")

    #     y_pred = model_xgb.predict(X_val.values)
    #     eval_df['y_pred'] = y_pred
    #     eval_dfs.append(eval_df)

    #     R2 = r2_score(y_val, y_pred)
    #     print(f"CV R2: {R2}")

    #     MSE = mean_squared_error(y_val, y_pred)
    #     RMSE = math.sqrt(MSE)
    #     print(f"CV RMSE: {RMSE}")
    #     cv_r2.append(R2)
    #     cv_rmse.append(RMSE)
    # model_name = "XGB"
    # record = "R2R"
    # trn_r2_mean = np.mean(trn_r2)
    # trn_rmse_mean = np.mean(trn_rmse)
    # cv_r2_mean = np.mean(cv_r2)
    # cv_rmse_mean = np.mean(cv_rmse)

    # print(f"train_r2_list: {trn_r2}")
    # print(f"cv_r2_list: {cv_r2}")
    # print(f"train_rmse_list: {trn_rmse}")
    # print(f"cv_rmse_list: {cv_rmse}")

    # print(f"train_r2: {trn_r2_mean}")
    # print(f"cv_r2: {cv_r2_mean}")
    # print(f"train_rmse: {trn_rmse_mean}")
    # print(f"cv_rmse: {cv_rmse_mean}")
    # # ~ ~ ~ Original code end ~ ~ ~

    # For training and testing metrics, create dataframes to store the results
    trn_r2 = pd.DataFrame(index=range(len(outer_cv)), columns=['train_r2'])
    trn_rmse = pd.DataFrame(index=range(len(outer_cv)), columns=['train_rmse'])
    cv_r2 = pd.DataFrame(index=range(len(outer_cv)), columns=['cv_r2'])
    cv_rmse = pd.DataFrame(index=range(len(outer_cv)), columns=['cv_rmse'])

    # To collect feature importance from each fold, create list of dataframes
    # (to be merged at the end, more efficient than appending to a df in each iteration)
    df_feat_imp = []

    # Generate the target variable and features
    # y is the column to be predicted, X is the rest of the data
    y = df_sampled[['aod']]
    X = df_sampled.drop(
        columns=['aod', 'date', 'grid_id', 'grid_id_50km', 'year_month'])

    # Loop through the outer cross-validation folds
    # For each fold, train the model and evaluate it on the validation set
    for n_fold, (trn_idx, val_idx) in enumerate(outer_cv):

        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        # Train the model using given hyperparameters
        # - n_jobs could also be defined from environment variable, e.g. SLURM_CPUS_PER_TASK
        # Note, that it would be possible to give evaluation metric(s) here, but in
        # then the metric would also be used for early stopping, which we don't want in this case.
        model_xgb = XGBRegressor(
            **best_params_XGB, n_jobs=1, tree_method=tree_method)
        model_xgb.fit(X_trn, y_trn.values.ravel())

        # Get the importances of features (for logging and analysis)
        df = pd.DataFrame({'feature': X_trn.columns, 'importance': model_xgb.feature_importances_, 'fold': n_fold}).sort_values(
            by=['importance'], ascending=False)
        df_feat_imp.append(df)

        # Predict on the training set and compute training metrics
        trn_y_pred = model_xgb.predict(X_trn)

        trn_r2.loc[n_fold] = r2_score(y_trn, trn_y_pred)
        trn_rmse.loc[n_fold] = math.sqrt(mean_squared_error(y_trn, trn_y_pred))

        # Predict on the validation set and compute validation metrics
        y_pred = model_xgb.predict(X_val.values)

        cv_r2.loc[n_fold] = r2_score(y_val, y_pred)
        cv_rmse.loc[n_fold] = math.sqrt(mean_squared_error(y_val, y_pred))

    # The final score of the cross validated model are the means of the scores
    # from all folds.
    trn_r2_mean = np.mean(trn_r2)
    trn_rmse_mean = np.mean(trn_rmse)
    cv_r2_mean = np.mean(cv_r2)
    cv_rmse_mean = np.mean(cv_rmse)

    print(f"Training R2: {trn_r2_mean}")
    print(f"Cross-Validation R2: {cv_r2_mean}")
    print(f"Training RMSE: {trn_rmse_mean}")
    print(f"Cross-Validation RMSE: {cv_rmse_mean}")

    # Diagnostics output: feature importances
    # (merge lists of df's to a single df)
    df_feat_imp = pd.concat(df_feat_imp, ignore_index=True)

    return model_xgb, {
        'train_r2': trn_r2_mean,
        'train_rmse': trn_rmse_mean,
        'cv_r2': cv_r2_mean,
        'cv_rmse': cv_rmse_mean,
        'feature_importance': df_feat_imp
    }


def evaluate_model(model, df_rest):
    """Evaluate the model on the rest of the data

    This function uses the trained model to predict the AOD values
    for the rest of the data (test set not used for training) and returns
    the predictions.

    Args:
        model (XGBRegressor): The trained XGBRegressor model.
        df_rest (pd.DataFrame): Dataframe with the rest of the data for evaluation.
    """

    # # ~ ~ ~ Original code ~ ~ ~
    # # # # # # # # # # # # # Prediction # # # # # # # # # # # # # #
    # rest_df['date'] = pd.to_datetime(rest_df['date'])
    # pd.set_option('display.max_rows', None)
    # print(df.isna().sum())
    # print(df.shape)
    # X_fin = rest_df[['aot_daily', 'co_daily', 'v_wind', 'u_wind', 'rainfall', 'temp',
    #                 'pressure', 'thermal_radiation', 'low_veg', 'high_veg', 'dewpoint_temp',
    #                 'elevation', 'water', 'shurub', 'urban', 'forest', 'savannas', 'month',
    #                 'day_of_year', 'cos_day_of_year', 'monsoon', 'lon', 'lat',
    #                 'wind_degree', 'RH', 'aot_rolling', 'co_rolling', 'omi_no2_rolling',
    #                 'v_wind_rolling', 'u_wind_rolling', 'rainfall_rolling', 'temp_rolling',
    #                 'wind_degree_rolling', 'RH_rolling', 'thermal_radiation_rolling',
    #                 'dewpoint_temp_rolling', 'aot_daily_annual', 'co_daily_annual',
    #                 'omi_no2_annual', 'v_wind_annual', 'u_wind_annual', 'rainfall_annual',
    #                 'thermal_radiation_annual', 'low_veg_annual', 'high_veg_annual',
    #                 'dewpoint_temp_annual', 'wind_degree_annual', 'RH_annual',
    #                 'co_daily_allyears']].copy()
    # pred = model_xgb.predict(X_fin)
    # if len(pred) != len(rest_df):
    #     raise ValueError(
    #         "Prediction length does not match the number of rows in rest_df")
    # rest_df['AOD_predicted'] = pred
    # rest_df.to_csv(
    #     "/scratch/users/akawano/pm_prediction/intermediate/ML_full_model/aod_validation_R2R.csv", index=False)
    # # ~ ~ ~ Original code end ~ ~ ~

    # Check if the model is trained
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model is not trained yet.")

    # Predict AOD values
    pred = model.predict(df_rest.drop(
        columns=['aod', 'date', 'grid_id', 'grid_id_50km', 'year_month']))

    # Check if prediction length matches the number of rows in rest_df
    if len(pred) != df_rest.shape[0]:
        raise ValueError(
            "Prediction length does not match the number of rows in rest_df")

    # Calculate metrics for evaluation
    r2 = r2_score(df_rest['aod'], pred)
    rmse = math.sqrt(mean_squared_error(df_rest['aod'], pred))

    print(f"Evaluation R2: {r2}")
    print(f"Evaluation RMSE: {rmse}")

    return {
        'r2': r2,
        'rmse': rmse
    }


if __name__ == "__main__":
    main()
