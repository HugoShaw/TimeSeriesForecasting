# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:59:13 2022

@author: Hugo Xue
@email: hugo@wustl.edu; 892849924@qq.com

"""

import time
import datetime
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from Windows import WindowGenerator
from dateutil.parser import parse
import pmdarima as pm
import warnings

class DNNForecast:
    def __init__(self, data, split_ratio=0.7, input_steps=12, output_steps=12, 
                 conv_width=3, max_epochs=20):
        """

        Parameters
        ----------
        data : royal canin transaction data
             if the data structure / labels change, it may cause problems
        split_ratio : train_test_split ratio, optional
            DESCRIPTION. The default is 0.7.
        input_steps : the historical input steps, optional
            It should be identical to output_steps. The default is 12 (12 periods).
        output_steps : the forecasting output steps, optional
            It should be identical to input_steps. The default is 12.
        conv_width : it is needed for CNN model, optional
            The convolutional width, see tensorflow doc. The default is 3.
        max_epochs : training maximal epochs, optional
            the maximal training epochs, see tensorflow doc. The default is 20.

        Returns
        -------
        Methods:
            fit(): training
            get_result(): return dataframe or dictionary
            get_avg_running_time(): average training time
            get_running_time(): total training time

        """
        self.data = data
        self.df = {"dc":[], "sku":[], "datetime":[], "true":[], "predict":[], 
                  "forecast_date":[], "forecast": [], "rSquared":[], "rmse":[],
                  "conf_int_999_lw":[],"conf_int_999_up":[],
                  "conf_int_995_lw":[],"conf_int_995_up":[],
                  "conf_int_99_lw":[],"conf_int_99_up":[],
                  "conf_int_95_lw":[], "conf_int_95_up":[], 
                  "conf_int_90_lw":[],"conf_int_90_up":[],
                  "conf_int_80_lw":[],"conf_int_80_up":[]}
        
        # train epochs
        self.MAX_EPOCHS = max_epochs
        #　use 3 months / 12 weeks of data to forecast one year data
        self.INPUT_STEPS = input_steps
        self.OUT_STEPS = output_steps
        self.CONV_WIDTH = conv_width # convolutional width
        self.split_ratio = split_ratio

        self.DAY = 24*60*60 # 24 hours * 60 minutes * 60 seconds
        self.WEEK = 7 * self.DAY # 7 days
        self.MONTH = 30 * self.DAY # a month
        self.YEAR = (365.2425) * self.DAY # a year
        
        # data inner structure 
        self.orderquantity = 'sum(gii__OrderQuantity__c)'
        self.orderdate = 'gii__OrderDate__c'
        self.warehouse = 'gii__Description__c'
        self.productSKU = 'giic_Product_SKU__c'
        self.DC = ['ASC- WHITTIER', 'ASC- MONROE', 'ASC- ATLANTA', 'ASC- FIFE',
               'ASC- DENVER', 'ASC- STOCKTON', 'ASC - BARTLETT', 'ASC- ORRVILLE',
               'ASC- SCHERTZ', 'ASC- ORLANDO']
        
        self.total_time = 0
        self.running_times = 0
    
    def get_result(self, option='dict'):
        """
        Parameters
        ----------
        option : 'dict' or 'dataframe', optional
            dict for dictionary, dataframe for pd. The default is 'dict'.

        Returns 
        -------
        get the training and predict result

        """
        if option == 'dict': return self.df
        elif option == 'dataframe': return pd.DataFrame(self.df)
    
    def get_avg_running_time(self):
        """
        get the average running time

        Returns
        -------
         avg run time (seconds)

        """
        print("==== average training time: {}====".format(self.total_time/self.running_times))
    
    def get_running_time(self):
        """
        get the total running time

        Returns
        -------
        total running time (seconds)

        """
        print("==== running time: {}====".format(self.total_time))
    
    def build_lstm_fit(self, window, num_features, patience=3):
        """
        
        Parameters
        ----------
        window : window class
            from window module, see tensorflow doc.
        num_features : the features numbers
            apply for multivariate time series dataset 
        patience : training patience time, optional
            see tensorflow doc. The default is 3.

        Returns
        -------
        history : training history
            see tensorflow doc.
        multi_lstm_model : lstm model
             see tensorflow doc.

        """
        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(64, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(self.OUT_STEPS*num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([self.OUT_STEPS, num_features])
        ])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                                        patience=patience,
                                                        mode='min')
    
        multi_lstm_model.compile(loss=tf.losses.MeanSquaredError(), # MeanSquaredError()
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])
    
        history = multi_lstm_model.fit(window.train, epochs=self.MAX_EPOCHS,
                          validation_data=window.val,
                          callbacks=[early_stopping])
        
        return history, multi_lstm_model
    
    def build_cnn_fit(self, window, num_features, patience=3):
        """
        build a convolutional neural network model

        Parameters
        ----------
        window : window class
            from window module, see tensorflow doc.
        num_features : the features numbers
            apply for multivariate time series dataset 
        patience : training patience time, optional
            see tensorflow doc. The default is 3.

        Returns
        -------
        history : training history
            see tensorflow doc.
        multi_cnn_model : cnn model
             see tensorflow doc.

        """
        multi_cnn_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.Lambda(lambda x: x[:, -self.CONV_WIDTH:, :]),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(self.CONV_WIDTH)),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Dense(self.OUT_STEPS*num_features,
                              kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([self.OUT_STEPS, num_features])
        ])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                                        patience=patience,
                                                        mode='min')
    
        multi_cnn_model.compile(loss=tf.losses.MeanSquaredError(), # MeanSquaredError()
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])
    
        history = multi_cnn_model.fit(window.train, epochs=self.MAX_EPOCHS,
                          validation_data=window.val,
                          callbacks=[early_stopping])
        
        return history, multi_cnn_model
    
    def R2_Score(self, y_true, y_hat):
        """
        Parameters
        ----------
        y_true : np.array
            true value of response
        y_hat : np.array
            predicted value of response
    
        Returns 
        -------
        r-squared 
    
        """    
        correlation_matrix = np.corrcoef(y_true, y_hat)
        correlation = correlation_matrix[0,1]
        rSquared = correlation**2
        
        return rSquared
    
    def RMSE_Score(self, y_true, y_hat):
        """
        Parameters
        ----------
        y_true : np.array
            true value of response
        y_hat : np.array
            predicted value of response
    
        Returns 
        -------
        rmse
    
        """ 
        return np.mean((y_true - y_hat)**2)**.5
        
    def fit(self, mode="lstm"):
        """
        training the forecasting model

        Parameters
        ----------
        mode : 'lstm' or 'cnn', optional
            lstm model or cnn model. The default is "lstm".

        Returns
        -------
        training result

        """
        start_time = time.time()
        for dc in self.DC:
            
            # Indicating it is training for dc
            print("===========Training Start: DC %s=================" %(dc))
            # each warehouse
            selected_dc = self.data[self.data[self.warehouse]==dc]
            SKUs = pd.unique(selected_dc[self.productSKU])
            print("==========There are %d SKUs in DC %s==========" %(len(SKUs),dc))
            
            # each fast moving sku data
            for sku in SKUs:
                try:
                    # print(len(selected_dc[selected_dc[self.productSKU] == sku]))
                    selected_sku = selected_dc[selected_dc[self.productSKU] == sku][[self.orderdate, self.orderquantity]]
            
                    wkly_data = selected_sku.resample('W-Mon', label='right', on=self.orderdate)\
                        .sum().reset_index().sort_values(by=self.orderdate)
                    
                    timestamp_s = wkly_data[self.orderdate].map(pd.Timestamp.timestamp)
                    wkly_data["Week_sin"] = np.sin(timestamp_s * (2 * np.pi / self.WEEK))
                    wkly_data["Week_cos"] = np.cos(timestamp_s * (2 * np.pi / self.WEEK))
                    wkly_data["Month_sin"] = np.sin(timestamp_s * (2 * np.pi / self.MONTH))
                    wkly_data["Month_cos"] = np.cos(timestamp_s * (2 * np.pi / self.MONTH))
                    wkly_data["Year_sin"] = np.sin(timestamp_s * (2 * np.pi / self.YEAR))
                    wkly_data["Year_cos"] = np.cos(timestamp_s * (2 * np.pi / self.YEAR))
                    date_index = wkly_data.pop(self.orderdate)
                    multi_fc_steps = pd.Series(
                        [date_index[len(date_index)-1]+datetime.timedelta(days=7) * (1+i) for i in range(self.OUT_STEPS)], 
                        index=[len(date_index)+i for i in range(self.OUT_STEPS)])
                    
            #         plt.plot(np.array(grp_skuDT["Year_sin"])[:365])
            #         print(grp_skuDT)
                    # print(wkly_data.info())
                    
                    # split the dataset into train and test 
                    n = len(wkly_data)
                    
                    train_df = wkly_data[0:int(n*self.split_ratio)]
                    val_df = wkly_data[int(n*self.split_ratio):]
                    
                    num_features = wkly_data.shape[1]
                
                    # window generator
                    w1 = WindowGenerator(input_width=self.INPUT_STEPS,
                                         label_width=self.OUT_STEPS,
                                         shift=self.OUT_STEPS,
                                         label_columns=[self.orderquantity],
                                         train_df=train_df,
                                         val_df=val_df,
                                         test_df=None)

                    # w2 for predict
                    w2 = WindowGenerator(input_width=self.INPUT_STEPS,
                                         label_width=self.INPUT_STEPS,
                                         shift=self.OUT_STEPS,
                                         label_columns=[self.orderquantity],
                                         train_df=wkly_data,
                                         val_df=None,
                                         test_df=None)
                    if mode == "cnn":
                        history, model = self.build_cnn_fit(w1, num_features)
                    elif mode == "lstm":
                        history, model = self.build_lstm_fit(w1, num_features)
                
                    multi_predict_y = model.predict(w2.predict, verbose=0)
                
                    prediction = []
                    for batch in multi_predict_y[:-1]:
                        prediction.append(batch[0][0])
                    last_batch = multi_predict_y[-1]
                    for _elem in last_batch:
                        prediction.append(_elem[0])    
                    
                    true = []
                    times=0
                    for _input, _label in w2.predict:
                        # print(_label.shape) # (32, x, 1)
                        for _elems in _label: # each batch (1, x, 1)
                            if times == 0: # first batch
                                for _elem in _elems: # each element [[0]]
                                    true.append(_elem.numpy()[0])
                            else: # not the first one
                                true.append(_elems.numpy()[-1][0])
                                
                            times+=1
                        
                    # # errors
                    _true = np.array(true[self.OUT_STEPS:])
                    _prediction = np.array(prediction[:-self.OUT_STEPS])

                    errors = _true - _prediction
                    err_std = np.std(errors)
                    err_mean = np.mean(errors)
                    
                    conf_int_999 = err_mean + err_std * 3.291
                    conf_int_995 = err_mean + err_std * 2.807
                    conf_int_99 = err_mean + err_std * 2.576
                    conf_int_95 = err_mean + err_std * 1.96
                    conf_int_90 = err_mean + err_std * 1.645
                    conf_int_80 = err_mean + err_std * 1.282

                    pred_up_999 = np.array(prediction) + conf_int_999
                    pred_lw_999 = np.array(prediction) - conf_int_999

                    pred_up_995 = np.array(prediction) + conf_int_995
                    pred_lw_995 = np.array(prediction) - conf_int_995

                    pred_up_99 = np.array(prediction) + conf_int_99
                    pred_lw_99 = np.array(prediction) - conf_int_99

                    pred_up_95 = np.array(prediction) + conf_int_95
                    pred_lw_95 = np.array(prediction) - conf_int_95

                    pred_up_90 = np.array(prediction) + conf_int_90
                    pred_lw_90 = np.array(prediction) - conf_int_90

                    pred_up_80 = np.array(prediction) + conf_int_80
                    pred_lw_80 = np.array(prediction) - conf_int_80

                    self.df["dc"].append(dc)
                    self.df["sku"].append(sku)
                    
                    started = 2 * self.OUT_STEPS
                    self.df["datetime"].append(date_index[started:].dt.strftime("%Y-%m-%d").tolist())
                    self.df["predict"].append(_prediction) # 12 periods ahead
                    self.df["true"].append(_true)
                    self.df["rSquared"].append(self.R2_Score(_true, _prediction))
                    self.df["rmse"].append(self.RMSE_Score(_true, _prediction))
                    
                    self.df["forecast_date"].append(multi_fc_steps)
                    self.df["forecast"].append(prediction[-self.OUT_STEPS:])
                    
                    self.df["conf_int_999_up"].append(pred_up_999)
                    self.df["conf_int_999_lw"].append(pred_lw_999)
                    self.df["conf_int_995_up"].append(pred_up_995)
                    self.df["conf_int_995_lw"].append(pred_lw_995)
                    self.df["conf_int_99_up"].append(pred_up_99)
                    self.df["conf_int_99_lw"].append(pred_lw_99)
                    self.df["conf_int_95_up"].append(pred_up_95)
                    self.df["conf_int_95_lw"].append(pred_lw_95)
                    self.df["conf_int_90_up"].append(pred_up_90)
                    self.df["conf_int_90_lw"].append(pred_lw_90)
                    self.df["conf_int_80_up"].append(pred_up_80)
                    self.df["conf_int_80_lw"].append(pred_lw_80)
                    
                    self.running_times+=1
                except:
                    self.df["dc"].append(dc)
                    self.df["sku"].append(sku)
                    self.df["datetime"].append([])
                    self.df["predict"].append([])
                    self.df["true"].append([])
                    self.df["rSquared"].append([])   
                    self.df["rmse"].append([])  
                    self.df["forecast_date"].append([])
                    self.df["forecast"].append([])
                    self.df["conf_int_999_up"].append([])
                    self.df["conf_int_999_lw"].append([])
                    self.df["conf_int_995_up"].append([])
                    self.df["conf_int_995_lw"].append([])
                    self.df["conf_int_99_up"].append([])
                    self.df["conf_int_99_lw"].append([])
                    self.df["conf_int_95_up"].append([])
                    self.df["conf_int_95_lw"].append([])
                    self.df["conf_int_90_up"].append([])
                    self.df["conf_int_90_lw"].append([])
                    self.df["conf_int_80_up"].append([])
                    self.df["conf_int_80_lw"].append([])
                    continue
        
        end_time = time.time()
        self.total_time = self.total_time + end_time - start_time
        
        
class ARIMA:
    def __init__(self, data, window_size=30):
        """

        Parameters
        ----------
        data : dataframe
            use exact data organization
        window_size : how many periods you need, optional
            predict based on how many periods. The default is 30.

        Returns
        -------
        None.

        """
        self.data = data
        self.window = window_size
        
        self.df = {"dc":[], "sku":[], "datetime":[], "true":[], 
                   "fc_datetime":[], "predict":[], 
                  "rSquared":[], "rmse":[],
                  "conf_int_999_lw":[],"conf_int_999_up":[],
                  "conf_int_995_lw":[],"conf_int_995_up":[],
                  "conf_int_99_lw":[],"conf_int_99_up":[],
                  "conf_int_95_lw":[], "conf_int_95_up":[], 
                  "conf_int_90_lw":[],"conf_int_90_up":[],
                  "conf_int_80_lw":[],"conf_int_80_up":[]}
        
        # data inner structure 
        self.orderquantity = 'sum(gii__OrderQuantity__c)'
        self.orderdate = 'gii__OrderDate__c'
        self.warehouse = 'gii__Description__c'
        self.productSKU = 'giic_Product_SKU__c'
        self.DC = ['ASC- WHITTIER', 'ASC- MONROE', 'ASC- ATLANTA', 'ASC- FIFE',
               'ASC- DENVER', 'ASC- STOCKTON', 'ASC - BARTLETT', 'ASC- ORRVILLE',
               'ASC- SCHERTZ', 'ASC- ORLANDO']
        
        self.total_time = 0
        self.running_times = 0
    
    def get_result(self, option='dict'):
        """
        Parameters
        ----------
        option : 'dict' or 'dataframe', optional
            dict for dictionary, dataframe for pd. The default is 'dict'.

        Returns 
        -------
        get the training and predict result

        """
        if option == 'dict': return self.df
        elif option == 'dataframe': return pd.DataFrame(self.df)
    
    def get_running_times(self):
        print("==== training times: {}====".format(self.running_times))
        
    def get_avg_running_time(self):
        """
        get the average running time

        Returns
        -------
         avg run time (seconds)

        """
        print("==== average training time: {}====".format(self.total_time/self.running_times))
    
    def get_running_time(self):
        """
        get the total running time

        Returns
        -------
        total running time (seconds)

        """
        print("==== running time: {}====".format(self.total_time))
    
    def R2_Score(self, y_true, y_hat):
        """
        Parameters
        ----------
        y_true : np.array
            true value of response
        y_hat : np.array
            predicted value of response
    
        Returns 
        -------
        r-squared 
    
        """    
        correlation_matrix = np.corrcoef(y_true, y_hat)
        correlation = correlation_matrix[0,1]
        rSquared = correlation**2
        
        return rSquared
    
    def RMSE_Score(self, y_true, y_hat):
        """
        Parameters
        ----------
        y_true : np.array
            true value of response
        y_hat : np.array
            predicted value of response
    
        Returns 
        -------
        rmse
    
        """ 
        return np.mean((y_true - y_hat)**2)**.5
    
    def fit(self):
        warnings.filterwarnings("ignore")

        start = time.time()
        for dc in self.DC:
            # Indicating it is training for dc
            print("Training Start: DC %s" %(dc))
            
            # each warehouse
            selected_dc = self.data[self.data[self.warehouse]==dc]
        
            # so far, try to predict all skus
        
            # fast moving sku: 
            SKUs = pd.unique(selected_dc[self.productSKU])
            
        #     metrics_sku = dict()
            print("==========There are %d SKUs in DC %s==========" %(len(SKUs),dc))
            
            for sku in SKUs:                
                try:
                    print("+++++++  Forecasting is progressing: %s +++++++" %(sku))
                    selected_sku = selected_dc[selected_dc[self.productSKU] == sku][[self.orderdate, self.orderquantity]]
                    
                    wkly_data = selected_sku.resample('W-Mon', label='right', on=self.orderdate)\
                        .sum().reset_index().sort_values(by=self.orderdate)
                    
                    wkly_data.set_index(self.orderdate, inplace=True)
                    
                    # Auto Model
                    totRows = len(wkly_data.index)
                    
                    sku_date_index = []
                    sku_true_result = []
                    
                    sku_fc_date_index = []
                    sku_forecast_result = []
                    
                    fc_999_up = []
                    fc_999_lw = []
                    fc_995_up = []
                    fc_995_lw = []
                    fc_99_lw = []
                    fc_99_up = []
                    fc_95_lw = []
                    fc_95_up = []
                    fc_90_lw = []
                    fc_90_up = []
                    fc_80_lw = []
                    fc_80_up = []
        
                    for i in range(self.window, totRows, 1):
                        trainset = wkly_data.iloc[:i,:]
                        
                        #  https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
                        autoModel = pm.auto_arima(trainset, start_p=1, start_q=1,
                                      test='adf',       # use adftest to find optimal 'd'
                                      max_p=10, max_q=10, # maximum p and q
                                      m=52,              # frequency of series
                                      d=None,           # let model determine 'd'
                                      seasonal=False,   # No Seasonality
                                      start_P=0, 
                                      D=0, 
                                      trace=False, #　A value of False will print no debugging information
                                      error_action='ignore',
                                      suppress_warnings=True, 
                                      stepwise=True)   
                        
                        fc = autoModel.predict(n_periods=1, return_conf_int=False)
                        fc_999 = autoModel.predict(n_periods=1, return_conf_int=True, alpha = 0.001)
                        fc_995 = autoModel.predict(n_periods=1, return_conf_int=True, alpha = 0.005)
                        fc_99 = autoModel.predict(n_periods=1, return_conf_int=True, alpha = 0.01)
                        fc_95 = autoModel.predict(n_periods=1, return_conf_int=True, alpha = 0.05)
                        fc_90 = autoModel.predict(n_periods=1, return_conf_int=True, alpha = 0.10)
                        fc_80 = autoModel.predict(n_periods=1, return_conf_int=True, alpha = 0.20)
                        
        
                        date_index = trainset.index[-1]
                        sku_date_index.append(date_index)
                        sku_true_result.append(trainset.values[-1][0])
                        # print(date_index, sku_date_index, trainset.values[-1])
                        
                        date_index += datetime.timedelta(days=7)
                        sku_fc_date_index.append(date_index)
                        sku_forecast_result.append(fc[-1])
                        # print(sku_fc_date_index, fc[-1])
    
                        fc_999_up.append(fc_999[1][0][0])
                        fc_999_lw.append(fc_999[1][0][1])
                        fc_995_up.append(fc_995[1][0][0])
                        fc_995_lw.append(fc_995[1][0][1])
                        fc_99_lw.append(fc_99[1][0][0])
                        fc_99_up.append(fc_99[1][0][1])
                        fc_95_lw.append(fc_95[1][0][0])
                        fc_95_up.append(fc_95[1][0][1])
                        fc_90_lw.append(fc_90[1][0][0])
                        fc_90_up.append(fc_90[1][0][1])
                        fc_80_lw.append(fc_80[1][0][0])
                        fc_80_up.append(fc_80[1][0][1])
                        # print(fc_999[1][0][0], fc_999[1][0][1])
                    # print(sku_true_result)
                    # print(sku_forecast_result)
                    self.df["dc"].append(dc)
                    self.df["sku"].append(sku)
                    self.df["datetime"].append(sku_date_index)
                    self.df["true"].append(sku_true_result)
                    
                    self.df["fc_datetime"].append(sku_fc_date_index)
                    self.df["predict"].append(sku_forecast_result)
                    
                    self.df["conf_int_999_up"].append(fc_999_up)
                    self.df["conf_int_999_lw"].append(fc_999_lw)
                    self.df["conf_int_995_up"].append(fc_995_up)
                    self.df["conf_int_995_lw"].append(fc_995_lw)
                    self.df["conf_int_99_up"].append(fc_99_up)
                    self.df["conf_int_99_lw"].append(fc_99_lw)
                    self.df["conf_int_95_up"].append(fc_95_up)
                    self.df["conf_int_95_lw"].append(fc_95_lw)
                    self.df["conf_int_90_up"].append(fc_90_up)
                    self.df["conf_int_90_lw"].append(fc_90_lw)
                    self.df["conf_int_80_up"].append(fc_80_up)
                    self.df["conf_int_80_lw"].append(fc_80_lw)
    
        
                    rsq = self.R2_Score(np.array(sku_true_result[1:]), np.array(sku_forecast_result[:-1]))
                    rmse = self.RMSE_Score(np.array(sku_true_result[1:]), np.array(sku_forecast_result[:-1]))
                    self.df["rSquared"].append(rsq)
                    self.df["rmse"].append(rmse)
                    
                    self.running_times+=1
                    # print("++++++++++  dc: %s | sku: %s  +++++++++++" %(dc, sku))
                
                except:
                    print("+++++++  Forecasting Failed: %s +++++++" %(sku))
                    self.df["dc"].append(dc)
                    self.df["sku"].append(sku)
                    self.df["datetime"].append([])
                    self.df["true"].append([])
                    
                    self.df["fc_datetime"].append([])
                    self.df["predict"].append([])
                    
                    self.df["conf_int_999_up"].append([])
                    self.df["conf_int_999_lw"].append([])
                    self.df["conf_int_995_up"].append([])
                    self.df["conf_int_995_lw"].append([])
                    self.df["conf_int_99_up"].append([])
                    self.df["conf_int_99_lw"].append([])
                    self.df["conf_int_95_up"].append([])
                    self.df["conf_int_95_lw"].append([])
                    self.df["conf_int_90_up"].append([])
                    self.df["conf_int_90_lw"].append([])
                    self.df["conf_int_80_up"].append([])
                    self.df["conf_int_80_lw"].append([])
                    
                    self.df["rSquared"].append([])
                    self.df["rmse"].append([])
                    continue
        
        end = time.time()
        self.total_time = self.total_time + (end-start)
        
        print("Processing time: %d" %(end-start))
        





