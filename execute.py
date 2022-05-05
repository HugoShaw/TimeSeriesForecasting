# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 13:00:13 2022

@author: Hugo Xue
@email: hugo@wustl.edu; 892849924@qq.com

"""
import pandas as pd
from Forecast import DNNForecast, ARIMA

# import data 
if __name__ == "__main__":
    data = pd.read_csv("./data/daily_transactions.csv", index_col=0,
                       parse_dates=['gii__OrderDate__c'])
    
    multi_steps_model = DNNForecast(data)
    multi_steps_model.fit(mode="lstm") # mode = 'cnn'
    df_result = multi_steps_model.get_result(option='dataframe') # option = 'dict'
    multi_steps_model.get_running_time() # total training time (s)
    multi_steps_model.get_avg_running_time() # the average training time (s)
    print(df_result.info())
    print(df_result.head())
    df_result.to_csv("./data/cnn_result.csv")
    
    arima_model = ARIMA(data, 90) # 90: use 90 days as a window
    arima_model.fit()
    df_result = arima_model.get_result(option = "dataframe")
    arima_model.get_running_times()
    arima_model.get_running_time()
    arima_model.get_avg_running_time()
    df_result.to_csv("./data/arima_result.csv")

# https://github.com/HugoShaw/TimeSeriesForecasting.git

# cnn:
# ==== running time: 13144.78158545494====
# ==== average training time: 2.3451885076636825====


























