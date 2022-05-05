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
   'ASC- SCHERTZ', 'ASC- DENVER'] #, 'ASC- ORLANDO'

# import forecasting data, real data, and the holdout data
lstm_data = pd.read_csv("./data/lstm_result.csv", index_col=0)
cnn_data = pd.read_csv("./data/cnn_result.csv", index_col=0)
arima = pd.read_csv("./data/arima_result.csv", index_col=0)
holdout_data = pd.read_csv("./data/daily_holdout.csv", index_col=0, parse_dates=[orderdate])

cnn_data.info()
lstm_data.info()
arima.info()
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

import re

################### lstm_cnn_arima_holdout ############################
dc_sku_dict = {}
for dc in DC:
    if dc not in dc_sku_dict:
        skus_in_holdout = pd.unique(
           holdout_data[holdout_data[warehouse] == dc][productSKU]).tolist()
        skus_in_pred = pd.unique(
           cnn_data[cnn_data["dc"] == dc]["sku"]).tolist()
        skus_in_lstm = pd.unique(
            lstm_data[lstm_data["dc"] == dc]["sku"]).tolist()
        skus_in_arima = pd.unique(
            arima[arima["dc"] == dc]["sku"]).tolist()
        skus_in_all = [sku for sku in skus_in_pred if sku in skus_in_holdout
                        and sku in skus_in_lstm and sku in skus_in_arima]
        dc_sku_dict[dc] = skus_in_all
    else:
        continue

all_dc_skus = []
for dc, skus in dc_sku_dict.items():
    for sku in skus:
        all_dc_skus.append((dc,sku))


index_list = [random.randint(0,len(all_dc_skus)) for i in range(1)]

selected_dc = all_dc_skus[index_list[0]][0]
selected_sku = all_dc_skus[index_list[0]][1]

cnn_rmse = cnn_data[(cnn_data["dc"] == selected_dc) & (cnn_data["sku"] == selected_sku)]["rmse"].values[0]
cnn_rsq = cnn_data[(cnn_data["dc"] == selected_dc) & (cnn_data["sku"] == selected_sku)]["rSquared"].values[0]

lstm_rmse = lstm_data[(lstm_data["dc"] == selected_dc) & (lstm_data["sku"] == selected_sku)]["rmse"].values[0]
lstm_rsq = lstm_data[(lstm_data["dc"] == selected_dc) & (lstm_data["sku"] == selected_sku)]["rSquared"].values[0]

arima_rmse = arima[(arima["dc"] == selected_dc) & (arima["sku"] == selected_sku)]["rmse"].values[0]
arima_rsq = arima[(arima["dc"] == selected_dc) & (arima["sku"] == selected_sku)]["rSquared"].values[0]

cnn_history_df, cnn_forecast_df = get_FC_data(cnn_data, selected_dc, selected_sku)
lstm_history_df, lstm_forecast_df = get_FC_data(lstm_data, selected_dc, selected_sku)

arima_data = arima[(arima["dc"] == selected_dc) & (arima["sku"] == selected_sku)]

for idx, row in arima_data.iterrows():
    dateData = row['datetime'] # 2021-01-25 | 2022-02-21
    dateData = re.findall(r"\d{4}-\d{2}-\d{2}", dateData)
        
    trueVal = row['true'].strip("[]").split(",")
    trueVal = [float(num) for num in trueVal if num != '']
        
    forecastDate = row['fc_datetime']
    forecastDate = re.findall(r"\d{4}-\d{2}-\d{2}", forecastDate)  

    forecastVal = row['predict'].strip("[]").split(",")
    forecastVal = [float(num) for num in trueVal if num != '']
    
holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)

dti = pd.date_range(start="2019-10-28", end="2022-02-28", freq='W-MON')
dti_arima_fc = pd.date_range(end="2022-03-07", periods=57, freq="W-MON")
dti_arima_hist = pd.date_range(end="2022-02-28", periods=57, freq="W-MON")

plt.style.use("seaborn")
plt.figure(figsize=(15,8))

# plt.plot(dti, cnn_history_df["true"],
#                       color='#ECBC5C', linestyle='--', label="History True")

# plt.plot(dti, cnn_history_df["predict"], 
#                       color='#394173', linestyle='-', label="History CNN")
# plt.plot(dti, lstm_history_df["predict"], 
#                       color='#51A7B7', linestyle='-', label="History LSTM")

plt.plot(cnn_forecast_df["date"], cnn_forecast_df["forecast"], 
                      color='#394173', marker='.', linestyle='dashed', label="Forcast CNN")
plt.plot(lstm_forecast_df["date"], lstm_forecast_df["forecast"], 
                      color='#51A7B7', marker='.', linestyle='dashed', label="Forecast LSTM")

# plt.plot(dti_arima_fc[:-1], forecastVal[:-1], 
#                       color='#A1B99D', marker='.', linestyle='-', label="History ARIMA")
plt.plot(dti_arima_fc[-1], forecastVal[-1], color="red", marker="o", label="Forecast ARIMA")

plt.plot(holdout_wkly_df[orderdate], holdout_wkly_df[orderquantity], 
                      color='#BC4030', linestyle='-.', label="Actual")

plt.title("DC: {0} | SKU: {1} | cnn_rmse: {2} | lstm_rmse: {3} | cnn_rsquared: {4} % | lstm_rsquared: {5} % | arima_rmse: {6} | arima_rsquared: {7} %".format(
selected_dc, selected_sku, round(cnn_rmse), round(lstm_rmse), round(cnn_rsq*100), round(lstm_rsq*100), round(float(arima_rmse)), round(float(arima_rsq)*100)))
plt.legend(loc='upper right', prop={'size':16})
plt.ylabel("weekly order quantity")
# plt.xlim("2021-01-01", "2022-07")

################################Time Consumption#############################
ys = np.array(["CNN", "LSTM", "ARIMA"])
heights = np.array([0.104, 0.146, 3.05])
plt.style.use("seaborn")
plt.figure(figsize=(15,8))

plt.barh(ys, heights, height=0.4, color=["#394173", "#A1B99D", "darkred"])

plt.xlabel("Days")

plt.title("Time Consumption Comparison Chart", fontdict={'fontsize':16})

# [3442528.8026451143, 3123.252094546356], 
#                     [0.18702004502007036, 0.28158597484362907]
############################ RMSE & R-Squared Comparison#############################
xs = np.array(["Royal Canin", "CNN"])

heights = np.array([3442528.8026451143, 3123.252094546356])

heights2 = np.array([0.18702004502007036, 0.28158597484362907])

df = pd.DataFrame({"RMSE": [3442528.8026451143, 3123.252094546356],
                   "R-Squared":[0.18702004502007036, 0.28158597484362907],
                   "Model": ["Royal Canin", "CNN"]})

plt.style.use("seaborn")

fig = plt.figure(figsize=(10,8))
# plt.show()
ax = fig.add_subplot(111)
ax2 = ax.twinx()

df.RMSE.plot(kind="bar", ax=ax, width=0.2, position=1, logy=True, color="darkred")
# df["R-Squared"].plot(kind="bar", ax=ax2, width=0.2, position=0)
df.plot(kind="line", ax=ax2, x="Model", y="R-Squared", linewidth=4, color="#394173", grid=False)

ax.set_ylabel("RMSE")
ax2.set_ylabel("R-Squared")

# ax.set_yscale("logit")
# ax.set_xlabel(["Royal Canin", "CNN"])
# ax2.set_xlabel("CNN")
# plt.grid(False)
plt.show()

plt.title("Time Consumption Comparison Chart", fontdict={'fontsize':16})

###############################################################################
xticks = [forecastDate[i] for i in range(len(forecastDate)) if i%4 == 0]

plt.style.use("seaborn")
plt.figure(figsize=(15,8))

plt.plot(dateData, trueVal, 
        color='#51A7B7', marker='.', linestyle='dashed', label="ARIMA_true")
plt.plot(forecastDate, forecastVal, 
        color='#BC4030', linestyle='-', label="ARIMA_predict")

##############################################################################
# plt.plot(holdout_wkly_df[orderdate][0], holdout_wkly_df[orderquantity][0], 
#                      color='red', marker="o", label="future_true")

plt.plot(forecastDate[-1], forecastVal[-1], color="red", marker="o", label="ARIMA_forecast")

plt.xticks(xticks)
plt.title("DC: {0} | SKU: {1} | arima_rmse: {2} % | arima_rsquared: {3} %".format(
selected_dc, selected_sku, round(float(arima_rmse)), round(float(arima_rsq)*100)))
plt.legend(loc='upper left')


# with open('../all_dc_forecast.json', 'rb') as fp:
#     all_dc_forecast = pickle.load(fp)

# data processing - clean data
arima.info()
# arima.head()
lstm_data.info()
cnn_data.info()
holdout_data.info()


# dc - sku dictionary
# rank based on rsquared

ranked_cnn_data = cnn_data.sort_values(by="rmse",axis=0,ascending=True)
ranked_cnn_data.dropna(subset=["rmse","rSquared"], inplace=True)

# what is the rmse distribution
# ranked_cnn_data["rmse"].plot(kind="bar")

dc_sku_dict = {}
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
all_dc_skus = []
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


result = pd.read_csv("result.csv", index_col=0, parse_dates=["datetime", "fc_datetime"])
result.info()
result.head()
plt.style.use("seaborn")
plt.figure(figsize=(15,8))

history_datetime = result["datetime"].dropna()
history_df = result['true'].dropna()
predict_df = result['predict'].dropna()
forecast_date = result['forecast_date'].dropna()
forecast = result['forecast'].dropna()

plt.plot(history_datetime, history_df, color='grey', linestyle='-', label="history_true")
plt.plot(history_datetime, predict_df, color='b', linestyle='-', label="history_predict")
plt.plot(forecast_date, forecast, color='red', marker='.', linestyle='dashed', label="future_predict_lstm")

plt.title("Order Quantity for the next week: {3} | r-Squared: {0} | rmse: {1}".format(
    result['rSquared'][0], result['rmse'][0]), forecast[0])
plt.legend(loc='upper left')
plt.xlabel("date")
plt.ylabel("order quantity")


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












#################################################################
cnn_data.head()
cnn_data.info()

cnn_forecast = {}
# get all predict values of cnn
for idx, row in cnn_data.iterrows():
    index = (row["dc"], row["sku"])
    
    fcData = row["forecast"].strip("[]").split(", ")
    fcData = [float(num) for num in fcData if num != '']
    
    cnn_forecast[index] = fcData
    
# get all true values of holdouts
# holdout_data.groupby([warehouse, productSKU]).resample('W-Mon', label='right', on=orderdate)\
#         .sum().reset_index().sort_values(by=orderdate)
holdout_forecast = {}
for key in cnn_forecast:
    try:
        selectedData = holdout_data[(holdout_data[warehouse]==key[0]) & (holdout_data[productSKU]==key[1])]
        selectedData = selectedData[[orderdate, orderquantity]]
        wkly_data = selectedData.resample('W-Mon', label='right', on=orderdate)\
            .sum().reset_index().sort_values(by=orderdate)
        holdout_forecast[key] = wkly_data
    except:
        continue


# get all predict values of original model
ori_pred = pd.read_excel('../rc.xlsx', sheet_name='Forecast', header=1)
# ori_pred.head()
# ASC - ATLANTA	902
# ASC - BARTLETT	904
# ASC - DENVER	906
# ASC - FIFE	907
# ASC - MONROE	911
# ASC - ORLANDO	903
# ASC - ORRVILLE	912
# ASC - SCHERTZ	905
# ASC - STOCKTON	908
# ASC - WHITTIER	909
code = {909:'ASC- WHITTIER', 911:'ASC- MONROE', 902:'ASC- ATLANTA', 
        907:'ASC- FIFE', 906:'ASC- DENVER', 908:'ASC- STOCKTON', 
        904:'ASC - BARTLETT', 912:'ASC- ORRVILLE', 905:'ASC- SCHERTZ'}

ori_pred[warehouse] = ori_pred["DC Code"].map(code)

ori_pred = ori_pred[[warehouse, "Item", 'W01/P03(W09) 2022',
                     'W02/P03(W10) 2022', 'W03/P03(W11) 2022',
                     'W04/P03(W12) 2022', 'W01/P04(W13) 2022']]

ori_forecast = {}
for idx, row in ori_pred.iterrows():
    index = (row[warehouse], str(row["Item"]))
    fcData = [row['W01/P03(W09) 2022'],row['W02/P03(W10) 2022'],
              row['W03/P03(W11) 2022'],row['W04/P03(W12) 2022'],
              row['W01/P04(W13) 2022']]
    ori_forecast[index] = fcData


# ori_pred.info()
# calculate the r-sqaured / rmse

# the metrics predict_df
from sklearn.metrics import mean_squared_error, r2_score
def my_r2_score(y_true, y_hat):
    y_bar = np.mean(y_true)
    ss_total = np.sum((y_true - y_bar) ** 2)
    ss_explained = np.sum((y_hat - y_bar) ** 2)
    ss_residual = np.sum((y_true - y_hat) ** 2)
    scikit_r2 = r2_score(y_true, y_hat)
    
    print(f'R-squared (SS_explained / SS_Total) = {ss_explained / ss_total}\n' + \
          f'R-squared (1 - (SS_residual / SS_Total)) = {1 - (ss_residual / ss_total)}\n'+ \
          f"Scikit-Learn's R-squared = {scikit_r2}")
     
def R2_Score(y_true, y_hat):
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

cnn_rmse = {}
cnn_rsquared = {}
for key in cnn_forecast:
    if key in holdout_forecast:
        try:
            y_pred = cnn_forecast[key][:5]
            y_true = holdout_forecast[key][orderquantity].tolist()
            cnn_rmse[key] = mean_squared_error(y_true, y_pred)
            cnn_rsquared[key] = R2_Score(np.array(y_true), np.array(y_pred))
        except:
            cnn_rsquared[key] = 0

ori_rmse = {}
ori_rsquared = {}
for key in ori_forecast:
    if key in holdout_forecast:
        try:
            y_pred = ori_forecast[key]
            y_true = holdout_forecast[key][orderquantity].tolist()
            ori_rmse[key] = mean_squared_error(y_true, y_pred)
            ori_rsquared[key] = R2_Score(np.array(y_true), np.array(y_pred))
        except:
            ori_rmse[key] = 0
            ori_rsquared[key] = 0
            
both_dic = {"cnn_rmse":[], "orig_rmse":[], 
            "cnn_rsquared":[], "orig_rsquared":[]}
for key in ori_forecast:
    if key in cnn_rmse:
        both_dic["cnn_rmse"].append(cnn_rmse[key])
        both_dic["orig_rmse"].append(ori_rmse[key])
    if key in cnn_rsquared:
        if np.isnan(cnn_rsquared[key]):
            both_dic["cnn_rsquared"].append(0)
        else:
            both_dic["cnn_rsquared"].append(cnn_rsquared[key])
        if np.isnan(ori_rsquared[key]):
            both_dic["orig_rsquared"].append(0)
        else:
            both_dic["orig_rsquared"].append(ori_rsquared[key])   

plt.style.use("seaborn")
plt.figure(figsize=(15,8))

plt.bar(x=[i for i in range(len(both_dic["cnn_rmse"]))], height=both_dic["cnn_rmse"], 
        color='#394173', linestyle='-', label="cnn model")
plt.bar(x=[i for i in range(len(both_dic["orig_rmse"]))], height=both_dic["orig_rmse"], 
        color='#BC4030', linestyle='-', label="royal canin model", alpha=0.5)
plt.axhline(y=np.mean(both_dic["cnn_rmse"]), label="average cnn rmse", color="darkblue")
plt.axhline(y=np.mean(both_dic["orig_rmse"]), label="average royal canin rmse", color="darkred")
# plt.yscale("log")
plt.legend()
plt.ylabel("Difference b/w forecasting result & true result​")
plt.xlabel("Forecasting result index​ ( Distribution Center – SKU )​")
plt.ylim(0, 1e7)



plt.style.use("seaborn")
plt.figure(figsize=(15,8))

plt.bar(x=[i for i in range(len(both_dic["cnn_rsquared"]))], height=both_dic["cnn_rsquared"], 
        color='#394173', linestyle='-', label="cnn model", alpha=0.3)
plt.bar(x=[i for i in range(len(both_dic["orig_rsquared"]))], height=both_dic["orig_rsquared"], 
        color='#BC4030', linestyle='-', label="royal canin model", alpha=0.3)
plt.axhline(y=np.mean(both_dic["cnn_rsquared"]), label="average cnn r-squared", color="darkblue")
plt.axhline(y=np.mean(both_dic["orig_rsquared"]), label="average royal canin r-squared", color="darkred")
plt.ylabel("Prediction Accuracy b/w forecasting result & true result​")
plt.xlabel("Forecasting result index​ ( Distribution Center – SKU )​")
plt.legend()



cnn_error = np.array(both_dic["cnn_rmse"])
ori_error = np.array(both_dic["orig_rmse"])

diff_err = (ori_error.mean() - cnn_error.mean())/(ori_error.mean())
diff_err.mean()

cnn_r2 = np.array(both_dic["cnn_rsquared"])
ori_r2 = np.array(both_dic["orig_rsquared"])

diff_err = (cnn_r2.mean()-ori_r2.mean())/(ori_r2.mean())
diff_err.mean()

# plt.title("Order Quantity for the next week: {3} | r-Squared: {0} | rmse: {1}".format(
#     result['rSquared'][0], result['rmse'][0]), forecast[0])
# plt.legend(loc='upper left')
# plt.xlabel("date")
# plt.ylabel("order quantity")














