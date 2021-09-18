from __future__ import division
import flask
import os
from flask import request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta,date

import warnings
warnings.filterwarnings("ignore")

import plotly.offline as pyoff
import plotly.graph_objs as go

import tensorflow.keras 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import statsmodels.formula.api as smf
#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.route('/api', methods=['GET'])
def index():
    #initiate plotly
    # pyoff.init_notebook_mode()
    #read the data in csv
    #df_sales = pd.read_csv('D:\Sales_Data.csv')
    #print()
    df_sales=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"\\Sales_Data.csv"))
    #convert date field from string to datetime
    df_sales['Date'] = pd.to_datetime(df_sales['Date'])
    #show first 10 rows
    df_sales.head(10)
    
    #groupby date and sum the sales
    df_sales = df_sales.groupby('Date').Amount.sum().reset_index()
    df_sales.head(10)

    #create a new dataframe to model the difference
    df_diff = df_sales.copy()
    #add previous sales to the next row
    df_diff['prev_sales'] = df_diff['Amount'].shift(1)
    #drop the null values and calculate the difference
    df_diff = df_diff.dropna()
    df_diff['diff'] = (df_diff['Amount'] - df_diff['prev_sales'])
    df_diff.head(10)

    #create dataframe for transformation from time series to supervised
    df_supervised = df_diff.drop(['prev_sales'],axis=1)
    #adding lags
    for inc in range(1,13):
        field_name = 'lag_' + str(inc)
        df_supervised[field_name] = df_supervised['diff'].shift(inc)
    #drop null values
    df_supervised = df_supervised.dropna().reset_index(drop=True)
    df_supervised.head(10)

    # Define the regression formula
    model = smf.ols(formula='diff ~ lag_1', data=df_supervised)
    # Fit the regression
    model_fit = model.fit()
    # Extract the adjusted r-squared
    regression_adj_rsq = model_fit.rsquared_adj

    df_model = df_supervised.drop(['Amount','Date'],axis=1)
    #split train and test set
    train_set, test_set = df_model[0:-6].values, df_model[-6:].values

    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)
    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False)

    y_pred = model.predict(X_test,batch_size=1)
    #for multistep prediction, you need to replace X_test values with the predictions coming from t-1

    #reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
    #rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0,len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
    #reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
    #inverse transform
    pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

    #create dataframe that shows the predicted sales
    result_list = []
    sales_dates = list(df_sales[-7:].Date)
    act_sales = list(df_sales[-7:].Amount)
    for index in range(0,len(pred_test_set_inverted)):
        result_dict = {}
        result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
        result_dict['Date'] = sales_dates[index+1]
        result_list.append(result_dict)
    # df_result = pd.DataFrame(result_list)
    print(jsonify(result_list))
    return jsonify(result_list) 


app.run()
