# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:45:51 2022

@author: Hugo Xue
@email: hugo@wustl.edu; 892849924@qq.com

"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from matplotlib import pyplot as plt
import time
import pickle
import warnings
import random
import seaborn as sns

# clarify variable names
orderquantity = 'sum(gii__OrderQuantity__c)'
orderdate = 'gii__OrderDate__c'
warehouse = 'gii__Description__c'
productSKU = 'giic_Product_SKU__c'
DC = ['ASC- FIFE', 'ASC- WHITTIER', 'ASC- STOCKTON',
  'ASC - BARTLETT', 'ASC- ORRVILLE', 'ASC- ATLANTA', 'ASC- MONROE',
   'ASC- SCHERTZ', 'ASC- DENVER', 'ASC- ORLANDO']

# import forecasting data, real data, and the holdout data
lstm_data = pd.read_csv("./data/lstm_result.csv", index_col=0)
cnn_data = pd.read_csv("./data/cnn_result.csv", index_col=0)
arima = pd.read_excel("./data/ARIMA_result.xlsx", index_col=0)
holdout_data = pd.read_csv("./data/daily_holdout.csv", index_col=0, parse_dates=[orderdate])

# with open('../all_dc_forecast.json', 'rb') as fp:
#     all_dc_forecast = pickle.load(fp)

# data processing - clean data
arima.info()
# arima.head()
lstm_data.info()
cnn_data.info()
holdout_data.info()

def get_FC_data(data, seleDC = "ASC- WHITTIER", seleSKU = '111103'):
    """
    extract data from DNN forecasting model result

    Parameters
    ----------
    data : dataframe
        the forecasting result, see the forecasting part
    seleDC : choose a distribution center, optional
        working on the result group by dc. The default is "ASC- WHITTIER".
    seleSKU : choose a sku code, optional
        working on the result group by sku. The default is '41006'.

    Returns
    -------
    a dataframe for a sku at dc

    """
    seleData = data[(data["dc"] == seleDC) & (data["sku"] == seleSKU)]
    
    # date
    date_series = seleData["datetime"].str.strip("[]").str.replace("'", "")
    date_list = date_series.str.split(", ").tolist()[0]
    # length = 123, starting from '2019-10-28' to '2022-02-28'
    date_series = pd.to_datetime(date_list) 
    
    # true
    true_list = seleData["true"].str.strip("[]").str.split(" ").tolist()[0]
    true_list = [num for num in true_list if num !=""]
    true_series = pd.Series([float(num.strip(".").strip(".\n")) for num in true_list])
    
    # prediction
    predict_list = seleData["predict"].str.strip("[]").str.split(" ").tolist()[0]
    predict_list = [num for num in predict_list if num !=""]
    predict_series = pd.Series([float(num.strip(".").strip(".\n")) for num in predict_list])
    
    # conf_int_95
    conf_int_95_lw_list = seleData["conf_int_95_lw"].str.strip("[]").str.split(" ").tolist()[0]
    conf_int_95_lw_list = [num for num in conf_int_95_lw_list if num !=""]
    conf_int_95_lw_series = pd.Series([float(num.strip(".").strip(".\n")) for num in conf_int_95_lw_list])
    
    conf_int_95_up_list = seleData["conf_int_95_up"].str.strip("[]").str.split(" ").tolist()[0]
    conf_int_95_up_list = [num for num in conf_int_95_up_list if num !=""]
    conf_int_95_up_series = pd.Series([float(num.strip(".").strip(".\n")) for num in conf_int_95_up_list])
    
    lengthOfpred = len(predict_series)
    
    history_df = pd.DataFrame({"date": date_series,
                               "true": true_series,
                               "predict": predict_series,
                               "conf_int_95_lw": conf_int_95_lw_series[:lengthOfpred],
                               "conf_int_95_up": conf_int_95_up_series[:lengthOfpred]})
    
    # forcast dataframe
    # time
    forecast_date = seleData["forecast_date"].str.findall("\d{4}-\d{2}-\d{2}").tolist()[0]
    forecast_date = pd.to_datetime(forecast_date)
    
    # value
    forecast_value = seleData["forecast"].str.strip("[]").str.split(", ").tolist()[0]
    forecast_value = pd.Series([float(num) for num in forecast_value])
    
    forecast_df = pd.DataFrame({"date": forecast_date,
                               "forecast": forecast_value,
                               "conf_int_95_lw": conf_int_95_lw_series[lengthOfpred:].values,
                               "conf_int_95_up": conf_int_95_up_series[lengthOfpred:].values})
    
    history_df["conf_int_95_lw"] = history_df["conf_int_95_lw"].apply(
        lambda x: 0 if x < 0 else x)
    history_df["predict"] = history_df["predict"].apply(
        lambda x: 0 if x < 0 else x)
    
    forecast_df["conf_int_95_lw"] = forecast_df["conf_int_95_lw"].apply(
        lambda x: 0 if x < 0 else x)
    forecast_df["forecast"] = forecast_df["forecast"].apply(
        lambda x: 0 if x < 0 else x)
    
    return history_df, forecast_df

def get_holdout_Data(data, seleDC = "ASC- WHITTIER", seleSKU = '111103'):
    seleData = data[(data[warehouse] == seleDC) & (data[productSKU] == seleSKU)]
    selectedData =seleData[[orderdate, orderquantity]]
    wkly_data = selectedData.resample('W-Mon', label='right', on=orderdate)\
        .sum().reset_index().sort_values(by=orderdate)
    
    return wkly_data

# dc - sku dictionary
# rank based on rsquared

ranked_cnn_data = cnn_data.sort_values(by="rmse",axis=0,ascending=True)
ranked_cnn_data.dropna(subset=["rmse","rSquared"], inplace=True)

# what is the rmse distribution
# ranked_cnn_data["rmse"].plot(kind="bar")

dc_sku_dict = {}
all_dc_skus = []
for dc in DC:
    if dc not in dc_sku_dict:
        skus_in_holdout = pd.unique(
           holdout_data[holdout_data[warehouse] == dc][productSKU]
           ).tolist()
        skus_in_pred = pd.unique(
           ranked_cnn_data[ranked_cnn_data["dc"] == dc]["sku"]
           ).tolist()
        skus_in_both = [sku for sku in skus_in_pred if sku in skus_in_holdout]
        dc_sku_dict[dc] = skus_in_both
    else:
        continue

for dc, skus in dc_sku_dict.items():
    for sku in skus:
        all_dc_skus.append((dc,sku))
# drop na
# ranked_cnn_data.dropna(subset=["rmse","rSquared"], inplace=True)
# 0 value is not possible
# ranked_cnn_data = ranked_cnn_data[(ranked_cnn_data["rmse"]!=0) & (ranked_cnn_data["rSquared"]!=0)]
ranked_cnn_data = cnn_data.sort_values(by="rmse",axis=0,ascending=True)
ranked_cnn_data.dropna(subset=["rmse","rSquared"], inplace=True)
ranked_cnn_data = ranked_cnn_data[(ranked_cnn_data["rmse"]!=0) & (ranked_cnn_data["rSquared"]!=0)]
ranked_cnn_data.reset_index(inplace=True)
# rmse distribution

# plt.ylabel("RMSE")
# plt.xlabel("SKU Index")

ranked_lstm_data = lstm_data.sort_values(by="rmse",axis=0,ascending=True)
ranked_lstm_data.dropna(subset=["rmse","rSquared"], inplace=True)
ranked_lstm_data = ranked_lstm_data[(ranked_lstm_data["rmse"]!=0) & (ranked_lstm_data["rSquared"]!=0)]
ranked_lstm_data.reset_index(inplace=True)
# rmse distribution
plt.plot(ranked_cnn_data["rmse"][:-2].values, color="r", label="CNN_RMSE")
plt.plot(ranked_lstm_data["rmse"][:-2].values, label="LSTM_RMSE")
plt.ylabel("RMSE")
plt.xlabel("SKU Index")

ranked_cnn_data = cnn_data.sort_values(by="rSquared",axis=0,ascending=False)
ranked_cnn_data.dropna(subset=["rmse","rSquared"], inplace=True)
ranked_cnn_data = ranked_cnn_data[(ranked_cnn_data["rmse"]!=0) & (ranked_cnn_data["rSquared"]!=0)]
# rmse distribution

ranked_lstm_data = lstm_data.sort_values(by="rSquared",axis=0,ascending=False)
ranked_lstm_data.dropna(subset=["rmse","rSquared"], inplace=True)
ranked_lstm_data = ranked_lstm_data[(ranked_lstm_data["rmse"]!=0) & (ranked_lstm_data["rSquared"]!=0)]
# rmse distribution
plt.plot(ranked_cnn_data["rSquared"].values, color="r", label="CNN R-Squared")
plt.plot(ranked_lstm_data["rSquared"].values, label="LSTM R-Squared")
plt.ylabel("R-Squared")
plt.xlabel("SKU Index")
# cnn_history_df, cnn_forecast_df = get_FC_data(cnn_data)
# cnn_forecast_df.head()
# display 8 charts
# ranked_cnn_data.reset_index(inplace=True)

def display(all_dc_skus, cnn_data, ranked_cnn_data, holdout_data, num = 8):
    index_list = [random.randint(0,len(all_dc_skus)) for i in range(num)]
    
    plt.style.use("seaborn")
    
    if num%2 == 0:
        x = num//2
        fig, axes = plt.subplots(num//2, 2, sharex=True, figsize=(20,10))
        for idx in range(num):
            selected_dc = all_dc_skus[index_list[idx]][0]
            selected_sku = all_dc_skus[index_list[idx]][1]
            rmse = cnn_data[(cnn_data["dc"] == selected_dc) & (cnn_data["sku"] == selected_sku)]["rmse"].values[0]
            rsq = cnn_data[(cnn_data["dc"] == selected_dc) & (cnn_data["sku"] == selected_sku)]["rSquared"].values[0]
            cnn_history_df, cnn_forecast_df = get_FC_data(ranked_cnn_data, selected_dc, selected_sku)
            holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)
            
            ax = axes[idx%x, idx//x]
            
            ax.plot(cnn_history_df["date"], cnn_history_df["true"], 
                                     color='green', linestyle='-', label="history_true")
            ax.plot(cnn_history_df["date"], cnn_history_df["predict"], 
                                     color='blue', linestyle='-', label="history_predict")
            ax.plot(cnn_forecast_df["date"], cnn_forecast_df["forecast"], 
                                     color='red', marker='.', linestyle='dashed', label="future_predict")
            ax.plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
                                     color='m', linestyle='-.', label="future_true")
            
            ax.set_title("DC: {0} | SKU: {1} | rmse: {2} | r-squared: {3} %".format(
                selected_dc, selected_sku, round(rmse), round(rsq*100)))
            ax.legend(loc='upper left')
    else:
        fig, axes = plt.subplots(num//2, 2, sharex=True, figsize=(20,10))
        for idx in range(8):
            selected_dc = all_dc_skus[index_list[idx]][0]
            selected_sku = all_dc_skus[index_list[idx]][1]
            rmse = cnn_data[(cnn_data["dc"] == selected_dc) & (cnn_data["sku"] == selected_sku)]["rmse"].values[0]
            rsq = cnn_data[(cnn_data["dc"] == selected_dc) & (cnn_data["sku"] == selected_sku)]["rSquared"].values[0]
            cnn_history_df, cnn_forecast_df = get_FC_data(ranked_cnn_data, selected_dc, selected_sku)
            holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)
            
            axes[idx%4, idx//4].plot(cnn_history_df["date"], cnn_history_df["true"], 
                                     color='green', linestyle='-', label="history_true")
            axes[idx%4, idx//4].plot(cnn_history_df["date"], cnn_history_df["predict"], 
                                     color='blue', linestyle='-', label="history_predict")
            axes[idx%4, idx//4].plot(cnn_forecast_df["date"], cnn_forecast_df["forecast"], 
                                     color='red', marker='.', linestyle='dashed', label="future_predict")
            axes[idx%4, idx//4].plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
                                     color='m', linestyle='-.', label="future_true")
            
            axes[idx%4, idx//4].set_title("DC: {0} | SKU: {1} | rmse: {2} | r-squared: {3} %".format(
                selected_dc, selected_sku, round(rmse), round(rsq*100)))
            axes[idx%4, idx//4].legend(loc='upper left')
    
    fig.set_tight_layout('rect')


index_list = [random.randint(0,len(all_dc_skus)) for i in range(8)]
# plt.subplots
# ggplot, seaborn, bmh, fivethirtyeight, dark_background, grayscale
plt.style.use("seaborn")
fig, axes = plt.subplots(4, 2, sharex=True, figsize=(20,10))

for idx in range(8):
    selected_dc = all_dc_skus[index_list[idx]][0]
    selected_sku = all_dc_skus[index_list[idx]][1]
    rmse = cnn_data[(cnn_data["dc"] == selected_dc) & (cnn_data["sku"] == selected_sku)]["rmse"].values[0]
    rsq = cnn_data[(cnn_data["dc"] == selected_dc) & (cnn_data["sku"] == selected_sku)]["rSquared"].values[0]
    cnn_history_df, cnn_forecast_df = get_FC_data(ranked_cnn_data, selected_dc, selected_sku)
    holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)
    
    axes[idx%4, idx//4].plot(cnn_history_df["date"], cnn_history_df["true"], 
                             color='green', linestyle='-', label="history_true")
    axes[idx%4, idx//4].plot(cnn_history_df["date"], cnn_history_df["predict"], 
                             color='blue', linestyle='-', label="history_predict")
    axes[idx%4, idx//4].plot(cnn_forecast_df["date"], cnn_forecast_df["forecast"], 
                             color='red', marker='.', linestyle='dashed', label="future_predict")
    axes[idx%4, idx//4].plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
                             color='m', linestyle='-.', label="future_true")
    
    axes[idx%4, idx//4].set_title("DC: {0} | SKU: {1} | rmse: {2} | r-squared: {3} %".format(
        selected_dc, selected_sku, round(rmse), round(rsq*100)))
    axes[idx%4, idx//4].legend(loc='upper left')

fig.set_tight_layout('rect')

index_list = [random.randint(0,len(all_dc_skus)) for i in range(8)]
# plt.subplots
fig, axes = plt.subplots(4, 2, sharex=True, figsize=(20,10))

for idx in range(8):
    selected_dc = all_dc_skus[index_list[idx]][0]
    selected_sku = all_dc_skus[index_list[idx]][1]
    rmse = lstm_data[(lstm_data["dc"] == selected_dc) & (lstm_data["sku"] == selected_sku)]["rmse"].values[0]
    rsq = lstm_data[(lstm_data["dc"] == selected_dc) & (lstm_data["sku"] == selected_sku)]["rSquared"].values[0]
    lstm_history_df, lstm_forecast_df = get_FC_data(lstm_data, selected_dc, selected_sku)
    holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)
    
    axes[idx%4, idx//4].plot(lstm_history_df["date"], lstm_history_df["true"], 
                             color='green', linestyle='-')
    axes[idx%4, idx//4].plot(lstm_history_df["date"], lstm_history_df["predict"], 
                             color='blue', linestyle='-')
    axes[idx%4, idx//4].plot(lstm_forecast_df["date"], lstm_forecast_df["forecast"], 
                             color='red', marker='.', linestyle='dashed')
    axes[idx%4, idx//4].plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
                             color='m', linestyle='-.')
    
    axes[idx%4, idx//4].set_title("DC: {0} | SKU: {1} | rmse: {2} | r-squared: {3} %".format(
        selected_dc, selected_sku, round(rmse), round(rsq*100)))
    axes[idx%4, idx//4].legend(loc='upper left')
    
fig.set_tight_layout('rect')





index_list = [random.randint(0,len(all_dc_skus)) for i in range(8)]

# plt.subplots
fig, axes = plt.subplots(4, 2, sharex=True, figsize=(20,10))
sns.set_theme()
for idx in range(8):
    selected_dc = all_dc_skus[index_list[idx]][0]
    selected_sku = all_dc_skus[index_list[idx]][1]
    cnn_history_df, cnn_forecast_df = get_FC_data(ranked_cnn_data, selected_dc, selected_sku)
    holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)
    
    ax = axes[idx%4, idx//4]
    
    sns.lineplot(data = cnn_history_df, x="date", y="true", ax=ax,legend=False)
    sns.lineplot(data = cnn_history_df, x="date", y="predict", ax=ax,legend=False, marker=".")
    sns.lineplot(data = cnn_forecast_df, x="date", y="forecast", ax=ax,legend=False, marker="+")
    p = sns.lineplot(data = holdout_wkly_df, x=orderdate, y=orderquantity, ax=ax,legend=False, marker="+")
    p.set(title="DC: {0} | SKU: {1}".format(selected_dc, selected_sku))
    p.set(xlabel="date", ylabel="order quantity")
    # p.legend(loc='upper right', labels=["history_true, history_predict, future_predict, future_true"])
    
    # axes[idx%4, idx//4].plot(cnn_history_df["date"], cnn_history_df["predict"], 
    #                          color='blue', linestyle='-')
    # axes[idx%4, idx//4].plot(cnn_forecast_df["date"], cnn_forecast_df["forecast"], 
    #                          color='red', marker='.', linestyle='dashed')
    # axes[idx%4, idx//4].plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
    #                          color='m', linestyle='-.')
    
    # sns.FacetGrid.set_titles("DC: {0} | SKU: {1}".format(selected_dc, selected_sku))

# fig.set_tight_layout('rect')




# ARIMA result
import datetime

index_list = [random.randint(0, len(arima)) for i in range(8)]
fig, axes = plt.subplots(4, 2, sharex=True, figsize=(20,10))
for idx in range(8):
    selected_dc = arima["dc"][index_list[idx]]
    selected_sku = arima["sku"][index_list[idx]]
    
    # generate a series of time
    endDate = arima["datetime"][index_list[idx]]
    dateLength = arima["forecast"][index_list[idx]]
    
    
    holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)
    
    ax = axes[idx%4, idx//4]
    
    ax.plot(lstm_history_df["date"], lstm_history_df["true"], 
                             color='green', linestyle='-')
    axes[idx%4, idx//4].plot(lstm_history_df["date"], lstm_history_df["predict"], 
                             color='blue', linestyle='-')
    axes[idx%4, idx//4].plot(lstm_forecast_df["date"], lstm_forecast_df["forecast"], 
                             color='red', marker='.', linestyle='dashed')
    axes[idx%4, idx//4].plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
                             color='m', linestyle='-.')
    
    axes[idx%4, idx//4].set_title("DC: {0} | SKU: {1} | rmse: {2} | r-squared: {3} %".format(
        selected_dc, selected_sku, round(rmse), round(rsq*100)))
    axes[idx%4, idx//4].legend(loc='upper left')
    
fig.set_tight_layout('rect')



import datetime
arima_clean = arima[arima["forecast"] != "[]"]
arima_clean.reset_index(inplace=True)
arima_dc_sku = []
for idx, row in arima_clean.iterrows():
    arima_dc_sku.append((row["dc"], row["sku"]))

inter_dc_sku = [skuSet for skuSet in arima_dc_sku if skuSet in all_dc_skus]

# arima.dropna(subset=["forecast"], inplace=True)
seleList = []
index_list = [random.randint(0, len(inter_dc_sku)) for i in range(8)]
plt.style.use("seaborn")
fig, axes = plt.subplots(4, 2, sharex=True, figsize=(20,10))
for idx in range(8):
    selected_dc = all_dc_skus[index_list[idx]][0]
    selected_sku = all_dc_skus[index_list[idx]][1]
    # print(selected_dc, selected_sku)
    seleList.append((selected_dc, selected_sku))
    
    data = arima[(arima["dc"] == selected_dc) & (arima["sku"] == selected_sku)]
    # generate a series of time
    endDate = data["datetime"]

    fcData = data["forecast"].str.strip("[]").str.split(", ").tolist()[0]
    fcData = [float(num) for num in fcData]
    
    conf_995_Data = data["conf_int_99_5_up"].str.strip("[]").str.split(", ").tolist()[0]
    conf_995_Data = [float(num) for num in conf_995_Data]
    
    fc_startDate = endDate - datetime.timedelta(days=7*len(fcData))
    
    # print(endDate, dateLength)
    fc_dti = pd.date_range(start=fc_startDate.values[0], end=endDate.values[0],
                        closed = "right", freq="W")
    
    trueData = data["true"].str.strip("[]").str.split(", ").tolist()[0]
    trueData = [float(num) for num in trueData]
    # print(len(fcData), len(trueData))
    
    # trueEndDate = endDate - datetime.timedelta(days=7)
    # trueStartDate = trueEndDate - datetime.timedelta(days=7*len(fcData))
    
    # true_dti = pd.date_range(start=trueStartDate.values[0], end=trueEndDate.values[0],
    #                 closed = "right", freq="W")
    
    holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)
    
    ax = axes[idx%4, idx//4]
    
    ax.plot(fc_dti, trueData, color='green', linestyle='-', label="true")
    ax.plot(fc_dti, fcData, color='blue', linestyle='-', label="forecast")
    
    # ax.plot(lstm_forecast_df["date"], lstm_forecast_df["forecast"], 
    #                          color='red', marker='.', linestyle='dashed')
    ax.plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
                              color='m', linestyle='-.', label="true_future")
    
    ax.plot(fc_dti[-1], fcData[-1], color="red", marker="o")
    ax.set_title("DC: {0} | SKU: {1}".format(
        selected_dc, selected_sku))
    ax.legend(loc='upper right')
fig.set_tight_layout('rect')
    

fig, axes = plt.subplots(4, 2, sharex=True, figsize=(20,10))

for idx in range(8):
    selected_dc = seleList[idx][0]
    selected_sku = seleList[idx][1]
    rmse = cnn_data[(cnn_data["dc"] == selected_dc) & (cnn_data["sku"] == selected_sku)]["rmse"].values[0]
    rsq = cnn_data[(cnn_data["dc"] == selected_dc) & (cnn_data["sku"] == selected_sku)]["rSquared"].values[0]
    cnn_history_df, cnn_forecast_df = get_FC_data(ranked_cnn_data, selected_dc, selected_sku)
    holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)
    
    axes[idx%4, idx//4].plot(cnn_history_df["date"], cnn_history_df["true"], 
                             color='green', linestyle='-', label="history_true")
    axes[idx%4, idx//4].plot(cnn_history_df["date"], cnn_history_df["predict"], 
                             color='blue', linestyle='-', label="history_predict")
    axes[idx%4, idx//4].plot(cnn_forecast_df["date"], cnn_forecast_df["forecast"], 
                             color='red', marker='.', linestyle='dashed', label="future_predict")
    axes[idx%4, idx//4].plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
                             color='m', linestyle='-.', label="future_true")
    
    axes[idx%4, idx//4].set_title("DC: {0} | SKU: {1} | rmse: {2} | r-squared: {3} %".format(
        selected_dc, selected_sku, round(rmse), round(rsq*100)))
    axes[idx%4, idx//4].legend(loc='upper left')

fig.set_tight_layout('rect')

fig, axes = plt.subplots(4, 2, sharex=True, figsize=(20,10))

for idx in range(8):
    selected_dc = seleList[idx][0]
    selected_sku = seleList[idx][1]
    rmse = lstm_data[(lstm_data["dc"] == selected_dc) & (lstm_data["sku"] == selected_sku)]["rmse"].values[0]
    rsq = lstm_data[(lstm_data["dc"] == selected_dc) & (lstm_data["sku"] == selected_sku)]["rSquared"].values[0]
    lstm_history_df, lstm_forecast_df = get_FC_data(lstm_data, selected_dc, selected_sku)
    holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)
    
    axes[idx%4, idx//4].plot(lstm_history_df["date"], lstm_history_df["true"], 
                             color='green', linestyle='-')
    axes[idx%4, idx//4].plot(lstm_history_df["date"], lstm_history_df["predict"], 
                             color='blue', linestyle='-')
    axes[idx%4, idx//4].plot(lstm_forecast_df["date"], lstm_forecast_df["forecast"], 
                             color='red', marker='.', linestyle='dashed')
    axes[idx%4, idx//4].plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
                             color='m', linestyle='-.')
    
    axes[idx%4, idx//4].set_title("DC: {0} | SKU: {1} | rmse: {2} | r-squared: {3} %".format(
        selected_dc, selected_sku, round(rmse), round(rsq*100)))
    axes[idx%4, idx//4].legend(loc='upper left')
    
fig.set_tight_layout('rect')



plt.figure(figsize=(15,8))

cnn_history_df, cnn_forecast_df = get_FC_data(ranked_cnn_data, selected_dc, selected_sku)
lstm_history_df, lstm_forecast_df = get_FC_data(lstm_data, selected_dc, selected_sku)
holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)

plt.plot(lstm_history_df["date"], lstm_history_df["true"], 
                         color='grey', linestyle='-', label="history_true")
plt.plot(lstm_history_df["date"], lstm_history_df["predict"], 
                         color='b', linestyle='-', label="history_predict_lstm")
plt.plot(lstm_forecast_df["date"], lstm_forecast_df["forecast"], 
                         color='orange', marker='.', linestyle='dashed',
                         label="future_predict_lstm")
plt.plot(cnn_history_df["date"], cnn_history_df["predict"], 
                             color='r', linestyle='-', label="history_predict_cnn")
plt.plot(cnn_forecast_df["date"], cnn_forecast_df["forecast"], 
                         color='g', marker='.', linestyle='dashed', label="future_predict_cnn")
plt.plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
                         color='c', linestyle='-.', label="future_true")

plt.title("DC: {0} | SKU: {1}".format(
    selected_dc, selected_sku))
plt.legend(loc='upper left')
plt.xlabel("date")
plt.ylabel("order quantity")
fig.set_tight_layout('rect')



















