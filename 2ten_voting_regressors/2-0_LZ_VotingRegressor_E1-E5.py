"""
--------------------------
File Name:  VotingRegressor.py
Contact: zhang (FZJ/IBG3) leojayak@gmail.com
Date: 09.03.23

Description: 
--------------------------
"""
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def format_val_file(file_val, index_col):
    """
    The columns is kid of different from the 'shuffled_20220317.csv'
    :param file_val: File for validate.
    :return: X_val and y_val
    """
    df_val = pd.read_csv(file_val, index_col=index_col)
    dataset2 = pd.to_datetime(df_val['Date'])
    DOY = dataset2.dt.dayofyear
    Year = dataset2.dt.year
    df_val.insert(0, 'DOY', DOY)
    df_val.insert(0, 'Year', Year)
    df_val.drop(labels=['Date', 'station', 'ESA-CCI', 'network'], axis=1, inplace=True)
    y_val = df_val.iloc[:, -2].values
    df_val.drop(labels='Soil Moisture', axis=1, inplace=True)
    X_val = df_val.iloc[:, :].values

    return X_val, y_val

def reg_VotingRegressor(reg_1, reg_2, reg_3, X_train, y_train, X_test, y_test, file_val, folder_independent,
                        folder_out=None, out_test=0):
    """
    :param reg_1: regressor a
    :param reg_2: regressor b
    :param reg_3: regressor c
    :param X_train: X_train
    :param y_train: y_train
    :param X_test: X_test
    :param y_test: y_test
    :param file_val: Validation dataset
    :param folder_independent: folder contains independent stations.
    :return: Trained VotingRegressor
    """
    # Train the VotingRegressor
    ereg = VotingRegressor(estimators=[('reg_1', reg_1), ('reg_2', reg_2), ('reg_3', reg_3)])
    ereg = ereg.fit(X_train, y_train)

    # ----------------------------
    # predict the test dataset.
    # ----------------------------
    pred = ereg.predict(X_test)
    pearson_r = pearsonr(y_test, pred)[0]
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    print('Testing: ')
    print(f'MSE: {mse}, RMSE: {rmse}, r2: {r2}, Pearson_r:, {pearson_r}')
    print('--------------- \n\n')

    if out_test ==1 :
        df_testRandom = pd.DataFrame(columns=['observed SSM', 'predicted SSM'])
        df_testRandom['observed SSM'] = pd.Series(data=y_test)
        df_testRandom['predicted SSM'] = pd.Series(data=pred)
        df_testRandom.to_csv(os.path.join(folder_out, 'test_random.csv'))


    # -----------------------
    # The validation dataset
    # ------------------------
    X_val, y_val = format_val_file(file_val, index_col=None)
    pred_val = ereg.predict(X_val)

    if out_test == 1:
        # Save the test temporal
        df_testTemporal = pd.DataFrame(columns=['observed SSM', 'predicted SSM'])
        df_testTemporal['observed SSM'] = pd.Series(data=y_val)
        df_testTemporal['predicted SSM'] = pd.Series(data=pred_val)
        df_testTemporal.to_csv(os.path.join(folder_out, 'test_temporal.csv'))

    df_val = pd.read_csv(file_val)
    stations_val = df_val['station'].unique()  # Get the station names

    # metrix for validation set.
    df_metrics_val = pd.DataFrame(columns=['station', 'mse', 'rmse', 'r2', 'pearson_r', 'NumberOfData'], dtype='object')

    for station in stations_val:

        # Get the index for data from 'station'
        idx = df_val.index[df_val['station'] == station]

        if len(idx) >= 2:
            S_metrics = pd.Series(index=['station', 'mse', 'rmse', 'r2', 'pearson_r', 'NumberOfData'], dtype='object')
            mse_idx = mean_squared_error(y_val[idx], pred_val[idx])
            rmse_idx = np.sqrt(mse_idx)
            r2_idx = r2_score(y_val[idx], pred_val[idx])
            pearson_r_idx = pearsonr(y_val[idx], pred_val[idx])[0]

            S_metrics['station'] = station
            S_metrics['mse'] = mse_idx
            S_metrics['rmse'] = rmse_idx
            S_metrics['r2'] = r2_idx
            S_metrics['pearson_r'] = pearson_r_idx
            S_metrics['NumberOfData'] = len(idx)
            del mse_idx, rmse_idx, r2_idx, pearson_r_idx

            df_metrics_val.loc[station] = S_metrics

    pred_val = ereg.predict(X_val)
    mse = mean_squared_error(y_val, pred_val)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, pred_val)
    pearson_r = pearsonr(y_val, pred_val)[0]
    print('Validation: ')
    print(f'MSE: {mse}, RMSE: {rmse}, r2: {r2}, Pearson_r:, {pearson_r}')
    print('--------------- \n\n')

    # -----------------------------
    # Independent stations
    # -----------------------------
    files = os.listdir(folder_independent)
    df_independent_metrics = pd.DataFrame(columns=['network', 'station', 'MSE', 'RMSE', 'R2', 'Pearson_r', 'n_size'],
                                          dtype='object')
    for idx, file in enumerate(files):
        # Read the data from independent stations.
        X_val_in, y_val_in = format_val_file(os.path.join(folder_independent, file), index_col=0)
        pred_val_in = ereg.predict(X_val_in)

        if out_test == 1:
            folder_independent = os.path.join(folder_out, 'independent_evaluation')
            if not os.path.exists(folder_independent):
                os.mkdir(folder_independent)
            # Save the estimation
            df_independent_station = pd.DataFrame(columns=['ID', 'observed SSM', 'predicted SSM'])
            df_independent_station['observed SSM'] = pd.Series(data=y_val_in)
            df_independent_station['predicted SSM'] = pd.Series(data=pred_val_in)
            df_independent_station.to_csv(os.path.join(folder_independent, file))




        mse = mean_squared_error(y_val_in, pred_val_in)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_in, pred_val_in)
        pearson_r = pearsonr(y_val_in, pred_val_in)[0]

        df_val_in = pd.read_csv(os.path.join(folder_independent, file))

        s_val_in = pd.Series(index=['network', 'station', 'MSE', 'RMSE', 'R2', 'Pearson_r', 'n_size'], dtype='object')
        s_val_in['network'] = file.split('_')[1]
        s_val_in['station'] = file.split('_')[2]
        s_val_in['MSE'] = mse
        s_val_in['RMSE'] = rmse
        s_val_in['R2'] = r2
        s_val_in['Pearson_r'] = pearson_r
        s_val_in['n_size'] = len(df_val_in)
        # print(idx, s_val_in)

        df_independent_metrics = df_independent_metrics.append(s_val_in, ignore_index=True)
        del s_val_in

    return df_independent_metrics, df_metrics_val


# ------------------------------
# work directory
# ------------------------------
work_dir = r'/data/private/ML_SSM/ML_SSM_dataset_v1_20220317'
os.chdir(work_dir)
file = r'ML_training&testing_v01shuffled_20220317.csv'
file_val = 'ML_validating_v01_20220303.csv'
folder_independent = 'output'


df = pd.read_csv(file)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# Split the training&testing data into training dataset and testing dataset, respectively.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

reg_AB = AdaBoostRegressor(n_estimators=30, learning_rate=0.2, loss='linear')
reg_GB = GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.5)
reg_KNR = KNeighborsRegressor(n_neighbors=4, weights='distance', p=1, leaf_size=20, algorithm='ball_tree')
reg_RFR = RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=4, min_samples_leaf=2)
reg_XB = XGBRegressor(n_estimator=800, max_depth=10, min_child_weight=1, gamma=0, subscample=0.8, colsample_bytree=0.9,
                      reg_alpha=0.05, reg_lambda=0.1)

# GB_KNR_RFR
folder_09 = '09_GB_KNR_RFR_output'
if not os.path.exists(folder_09):
    os.mkdir(folder_09)
df_GB_KNR_RFR, df_GB_KNR_RFR_val = reg_VotingRegressor(reg_GB, reg_KNR, reg_RFR, X_train, y_train, X_test, y_test,
                                                       file_val, folder_independent,
                                                       folder_out=folder_09, out_test=1)
