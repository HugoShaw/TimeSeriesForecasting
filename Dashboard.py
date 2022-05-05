# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:52:24 2022

@author: Hugo Xue
@email: hugo@wustl.edu; 892849924@qq.com

"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from matplotlib import pyplot as plt
import time
from dash import Dash, dcc, html, Input, Output
import pickle
import plotly.express as px
import plotly.graph_objects as go
import warnings

# df = {"dc":[], "sku":[], "datetime":[], "true":[], "predict":[], 
#           "forecast_date":[], "forecast": [], "rSquared":[], "rmse":[],
#           "conf_int_999_lw":[],"conf_int_999_up":[],
#           "conf_int_995_lw":[],"conf_int_995_up":[],
#           "conf_int_99_lw":[],"conf_int_99_up":[],
#           "conf_int_95_lw":[], "conf_int_95_up":[], 
#           "conf_int_90_lw":[],"conf_int_90_up":[],
#           "conf_int_80_lw":[],"conf_int_80_up":[]}

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
holdout_data = pd.read_csv("./data/daily_holdout.csv", index_col=0, parse_dates=[orderdate])

# data processing - clean data
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
# what is the rmse distribution
ranked_cnn_data["rmse"].plot(kind="bar")


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

windows=60
# cnn forecast
# cnn_onestep_forecast_result = pd.read_excel("cnn_onestep_forecast_result.xlsx")



# with open('all_dc_forecast.json', 'rb') as fp:
#     all_dc_forecast = pickle.load(fp)
# coordinates = {'ASC- FIFE': (47.2324, 122.3474), 
#                'ASC- WHITTIER': (33.98, -118.07), 
#                'ASC- STOCKTON': (37.9009, 121.4228),
#                'ASC - BARTLETT': (41.9753, 88.2169), 
#                'ASC- ORRVILLE': (40.8455, 81.7568), 
#                'ASC- ATLANTA': (34.0670, 84.0907), 
#                'ASC- MONROE': (40.3363, 74.4330),
#                'ASC- SCHERTZ': (29.5886, 98.2760),
#                'ASC- SCHERTZ': (29.5886, 98.2760), 
#                'ASC- DENVER': (39.7781, 104.8747)}
dict_loc = {'ASC- FIFE': ['WA'], 
                'ASC- WHITTIER': ['CA'], 
                'ASC- STOCKTON': ['CA'],
                'ASC - BARTLETT': ['IL'], 
                'ASC- ORRVILLE':['OH'], 
                'ASC- ATLANTA':['GA'], 
                'ASC- MONROE': ['NJ'],
                'ASC- SCHERTZ': ['TX'],
                'ASC- DENVER':['CO']}

# cnn_onestep_forecast_result["coordinates"] = cnn_onestep_forecast_result["dc"].apply(
#     lambda x: coordinates[x])
# ===dc: ASC- WHITTIER====
# ====zip code: [90606]===== 33.98, -118.07
# ===dc: ASC- MONROE====
# ====zip code: [8831]===== 40.3363° N, 74.4330° W
# ===dc: ASC- ATLANTA====
# ====zip code: [30024]===== 34.0670° N, 84.0907° W
# ===dc: ASC- FIFE====
# ====zip code: [98424]=====47.2324° N, 122.3474° W
# ===dc: ASC- DENVER====
# ====zip code: [80238]===== 39.7781° N, 104.8747° W
# ===dc: ASC- STOCKTON====
# ====zip code: [95206]===== 37.9009° N, 121.4228° W
# ===dc: ASC - BARTLETT====
# ====zip code: [60103]===== 41.9753° N, 88.2169° W
# ===dc: ASC- ORRVILLE====
# ====zip code: [44667]===== 40.8455° N, 81.7568° W
# ===dc: ASC- SCHERTZ====
# ====zip code: [78154]===== 29.5886° N, 98.2760° W
# ===dc: ASC- ORLANDO====
# ====zip code: [32809]===== 28.4589° N, 81.3949° W

tab_style = {"background":"darkred",
            "color":"white",
            "text-transform":"uppercase",
            "justify-content":"center",
            "border":"grey",
            "border-radius":"10px",
            "font-size":"12px",
            "font-weight":600,
            "align-items":"center",
            "padding":"12px"}

tab_selected_style = {"background": "darkblue",
                      'color': 'white',
                      'text-transform': 'uppercase',
                      'justify-content': 'center',
                      'border-radius': '10px',
                      'font-weight': 600,
                      'font-size': '12px',
                      'align-items': 'center',
                      'padding':'12px'}

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Royal Canin Forecasting Dashboard", 
            style={"textAlign":"center", "color":"darkred"}),
    
    html.Hr(),
    html.Div(id='metric-benchmark'),
    html.Hr(),
    
    # tab to change the charts
    html.Div([
        dcc.Tabs(id="Chart-Tabs",
                 value="Chart_ARIMA",
                 children=[
                     dcc.Tab(label="ARIMA",
                             value="ARIMA",
                             style=tab_style,
                             selected_style=tab_selected_style),
                     dcc.Tab(label="LSTM",
                             value="LSTM",
                             style=tab_style,
                             selected_style=tab_selected_style),
                     dcc.Tab(label="CNN",
                             value="CNN",
                             style=tab_style,
                             selected_style=tab_selected_style)
                     ]),
    ]),
    
    # drop down container
    html.Div([
        # dropdown cell distribution center
        html.Div([
            html.Label("Distribution Center: "),   
            dcc.Dropdown(                
                id="dc-column",
                options = [{'label': dc, 'value': dc} for dc in dc_sku_dict.keys()],
                value = 'ASC- DENVER'
            )
        ], style={'width':'48%', 'display':'inline-block'}),
        
        # TODO: add authentical skus / use radio to control
        # dropdown cell skus
        html.Div([
            html.Label("SKU: "),
            dcc.Dropdown(id='SKU-dropdown')
        ], style={'width':'48%', 'display':'inline-block', 'float':'right'})
    ], style={"padding":'10px 5px'}),

    html.Hr(),
    
    # graph output
    html.Div([
        # geography mapping
        html.Div([dcc.Graph(id="map-output")],
                  style={'display': 'inline-block', 'width': '30%'}),
        
        # figs
        html.Div([dcc.Graph(id="figure-output")],
            style={'width':'70%', 'display':'inline-block', 'float':'right'})
    ], style={"padding":'10px 5px'})
    
    # # dcc.Slider(
    # #     )

    # html.Div([html.Label("Convolutional Neural Network Forecasting Result"),
    #           dcc.Graph(id="cnn-output")])
    
    # html.Div([html.Label("Long Term Short Memory Forecasting Result"),
    #           dcc.Graph(id="lstm-output")])
    
])

@app.callback(
    Output(component_id="SKU-dropdown",component_property="options"),
    [Input(component_id="dc-column",component_property="value")]
)
def set_sku_options(selected_dc):
    return [{"label": sku, "value":sku} for sku in dc_sku_dict[selected_dc]]

@app.callback(
    Output(component_id="SKU-dropdown",component_property="value"),
    [Input(component_id="SKU-dropdown",component_property="options")]
)
def set_sku_value(available_options):
    return available_options[0]['value']

@app.callback(
    Output(component_id="metric-benchmark",component_property="children"),
    [Input(component_id="dc-column",component_property="value"),
      Input(component_id="SKU-dropdown",component_property="value"),
      Input(component_id='Chart-Tabs',component_property='value')])
def set_metrics_value(selected_dc, selected_sku, selected_fig):
    lstm_history_df, lstm_forecast_df = get_FC_data(lstm_data, selected_dc, selected_sku)
    cnn_history_df, cnn_forecast_df = get_FC_data(cnn_data, selected_dc, selected_sku)
    if selected_fig == "ARIMA":
        pass
    elif selected_fig == "LSTM":
        nextWk = lstm_forecast_df["forecast"][0]
        rsq = lstm_data[((lstm_data["dc"]==selected_dc) & (lstm_data["sku"]==selected_sku))]["rmse"].values
        return html.Div([
            html.H3(f"LSTM-Next Week Order Quantity: {nextWk}  |  RMSE: {rsq}",
                    style={"color": "darkgreen"})
            ])
    else:
        nextWk = cnn_forecast_df["forecast"][0]
        rsq = cnn_data[((cnn_data["dc"]==selected_dc) & (cnn_data["sku"]==selected_sku))]["rmse"].values
        return html.Div([
            html.H3(f"CNN-Next Week Order Quantity: {nextWk}  |  RMSE: {rsq}",
                    style={"color": "darkgreen"})
            ])

@app.callback(
    Output(component_id="map-output", component_property="figure"),
    Output(component_id="figure-output", component_property="figure"),
    [Input(component_id="dc-column", component_property="value"),
      Input(component_id="SKU-dropdown", component_property="value"),
      Input(component_id='Chart-Tabs',component_property='value')]
)
def build_fc_figs(selected_dc, selected_sku, selected_fig):
    # load data
    lstm_history_df, lstm_forecast_df = get_FC_data(lstm_data, selected_dc, selected_sku)
    cnn_history_df, cnn_forecast_df = get_FC_data(cnn_data, selected_dc, selected_sku)
    holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)
    
    colors = ["<10","<50","<100","<500","<5000"]
    
    # line-chart-arima
    if selected_fig == "ARIMA":
        fig1 = px.choropleth()
        
        fig2 = go.Figure()
        
        return fig1, fig2
    elif selected_fig == "LSTM":
        # the first forecast result 
        first_period_res = lstm_forecast_df["forecast"].values[0]
        thres = float(first_period_res)
        if thres < 10:
            # lim = limits[0]
            c = colors[0]
        elif thres < 50:
            # lim = limits[1]
            c = colors[1]
        elif thres < 100:
            # lim = limits[2]
            c = colors[2]
        elif thres < 500:
            # lim = limits[3]
            c = colors[3]
        elif thres < 5000:
            # lim = limits[4]
            c = colors[4]
        
        fig3 = px.choropleth(locations=dict_loc[selected_dc],
                        locationmode="USA-states",
                        color=[c], scope="usa")
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Scatter(x=lstm_history_df["date"],
                          y=lstm_history_df["true"],
                        name="true order quantity",
                        line=dict(color='rgba(102,166,30,0.7)',width=2)))

        fig4.add_trace(go.Scatter(x=lstm_history_df["date"],
                                  y=lstm_history_df["predict"], 
                                name="predicted order quantity",
                                line=dict(color='rgba(228,26,28,1)',width=2)))
    
        fig4.add_trace(go.Scatter(x=lstm_forecast_df["date"], 
                                  y=lstm_forecast_df["forecast"], 
                                name="Forecasted order quantity",
                                line=dict(color='rgba(200,20,20,1)',width=2)))
    
        fig4.add_trace(go.Scatter(x=holdout_wkly_df[orderdate], 
                                  y=holdout_wkly_df[orderquantity], 
                                name="Real Order Quantity",
                                line=dict(color='rgba(100,100,20,1)',width=2)))
        
        return fig3, fig4
    else:
        # the first forecast result 
        first_period_res = cnn_forecast_df["forecast"].values[0]
        thres = float(first_period_res)
        if thres < 10:
            # lim = limits[0]
            c = colors[0]
        elif thres < 50:
            # lim = limits[1]
            c = colors[1]
        elif thres < 100:
            # lim = limits[2]
            c = colors[2]
        elif thres < 500:
            # lim = limits[3]
            c = colors[3]
        elif thres < 5000:
            # lim = limits[4]
            c = colors[4]
        
        fig5 = px.choropleth(locations=dict_loc[selected_dc],
                        locationmode="USA-states",
                        color=[c], scope="usa")
        
        fig6 = go.Figure()
        
        fig6.add_trace(go.Scatter(x=cnn_history_df["date"],
                          y=cnn_history_df["true"],
                        name="true order quantity",
                        line=dict(color='rgba(102,166,30,0.7)',width=2)))

        fig6.add_trace(go.Scatter(x=cnn_history_df["date"],
                                  y=cnn_history_df["predict"], 
                                name="predicted order quantity",
                                line=dict(color='rgba(228,26,28,1)',width=2)))
    
        fig6.add_trace(go.Scatter(x=cnn_forecast_df["date"], 
                                  y=cnn_forecast_df["forecast"], 
                                name="Forecasted order quantity",
                                line=dict(color='rgba(200,20,20,1)',width=2)))
    
        fig6.add_trace(go.Scatter(x=holdout_wkly_df[orderdate], 
                                  y=holdout_wkly_df[orderquantity], 
                                name="Real Order Quantity",
                                line=dict(color='rgba(100,100,20,1)',width=2)))
        
        return fig5, fig6
    
# @app.callback(
#     Output(component_id="figure-output",component_property="figure"),
#     [Input(component_id="dc-column",component_property="value"),
#      Input(component_id="SKU-dropdown",component_property="value")]
# )
# def display_foreacast(selected_dc, selected_sku):
#     dt = data[data["Primary Warehouse"]==selected_dc]
#     skuDT = dt[dt["Product SKU"] == selected_sku]
#     grp_skuDT = skuDT.groupby(["Order Date"])["Order Quantity"].agg("sum")
#     grp_skuDT = grp_skuDT.reset_index()
#     grp_skuDT.set_index("Order Date", inplace=True)
#     grp_skuDT = grp_skuDT.asfreq(freq='d',fill_value=0)
#     grp_skuDT = grp_skuDT.reset_index()
#     wk_skuDT = grp_skuDT.resample('W-Mon',label='right',closed='right',on='Order Date')\
#     .sum().reset_index().sort_values(by='Order Date')
#     wk_skuDT.set_index("Order Date", inplace=True)
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=wk_skuDT["Order Quantity"].index, 
#                              y=wk_skuDT["Order Quantity"].values, 
#                             name="training",
#                             line=dict(color='rgba(102,166,30,0.7)',width=2)))
    
#     fig.add_trace(go.Scatter(x=wk_skuDT.iloc[windows:,0].index, 
#                              y=wk_skuDT.iloc[windows:,0].values, 
#                             name="testing(actual)",
#                             line=dict(color='rgba(228,26,28,1)',width=2)))
    
#     forecast_data = pd.Series(all_dc_forecast[selected_dc][selected_sku])
    
#     fig.add_trace(go.Scatter(x=forecast_data.index, 
#                              y=forecast_data.values, 
#                             name="forecasting",
#                             line=dict(color='rgba(55,126,184,1)',width=2)))

#     return fig

# @app.callback(
#     Output(component_id="map-output",component_property="figure"),
#     [Input(component_id="dc-column",component_property="value"),
#      Input(component_id="SKU-dropdown",component_property="value")]
# )
# def display_map(selected_dc, selected_sku):
#     # display new forecast day in the following weeks on map
#     data = cnn_onestep_forecast_result[
#         cnn_onestep_forecast_result["dc"]==selected_dc]
#     data = data[data["sku"]==selected_sku]
#     # dc_coordinates = coordinates[selected_dc]
#     y_forecast = data["forecast"].tolist()[0].replace("[","").replace("]","")\
#         .replace(" ","").split(",")[-1]
    
#     # limits = [(0,10),(10,50),(50,100),(100,500),(500, 5000)]
#     colors = ["<10","<50","<100","<500","<5000"]
    
#     # scale = 1
#     thres = float(y_forecast)
#     if thres < 10:
#         # lim = limits[0]
#         c = colors[0]
#     elif thres < 50:
#         # lim = limits[1]
#         c = colors[1]
#     elif thres < 100:
#         # lim = limits[2]
#         c = colors[2]
#     elif thres < 500:
#         # lim = limits[3]
#         c = colors[3]
#     elif thres < 5000:
#         # lim = limits[4]
#         c = colors[4]
     
#     fig = px.choropleth(locations=dict_loc[selected_dc],
#                         locationmode="USA-states", 
#                         color=[c], scope="usa")

#     return fig

# @app.callback(
#     Output(component_id="cnn-output",component_property="figure"),
#     [Input(component_id="dc-column",component_property="value"),
#      Input(component_id="SKU-dropdown",component_property="value")]
# )
# def cnn_forecast(selected_dc, selected_sku):
#     data = cnn_onestep_forecast_result[
#            cnn_onestep_forecast_result["dc"]==selected_dc]
#     data = data[data["sku"]==selected_sku]
#     time_index = data["datetime"].tolist()[0].replace("[","").replace("]","")\
#         .replace("\n", "").replace("T00:00:00.000000000","").split(" ")
#     x_index = pd.to_datetime(time_index)
#     y_forecast = pd.Series(data["forecast"].tolist()[0].replace("[","").replace("]","")\
#         .replace(" ","").split(","), dtype='float64')
#     y_true = pd.Series(data["true"].tolist()[0].replace("[","").replace("]","")\
#         .replace(" ","").split(","), dtype='float64')
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=x_index[:-1],
#                              y=y_true, 
#                             name="true order quantity",
#                             line=dict(color='rgba(102,166,30,0.7)',width=2)))
    
#     fig.add_trace(go.Scatter(x=x_index, 
#                              y=y_forecast, 
#                             name="estimated order quantity",
#                             line=dict(color='rgba(228,26,28,1)',width=2)))
    
#     fig.update_layout(
#         title="For the next week, the forecasted order quantity is {}".format(y_forecast[len(y_forecast)-1]))
    

#     return fig
    
# @app.callback(
#     Output(component_id="lstm-output",component_property="figure"),
#     [Input(component_id="dc-column",component_property="value"),
#       Input(component_id="SKU-dropdown",component_property="value")]
# )
# def lstm_forecast(selected_dc, selected_sku):
#     lstm_history_df, lstm_forecast_df = get_FC_data(lstm_data, selected_dc, selected_sku)
#     holdout_wkly_df = get_holdout_Data(holdout_data, selected_dc, selected_sku)   
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=lstm_history_df["date"],
#                               y=lstm_history_df["true"], 
#                             name="true order quantity",
#                             line=dict(color='rgba(102,166,30,0.7)',width=2)))

#     fig.add_trace(go.Scatter(x=lstm_history_df["date"],
#                               y=lstm_history_df["predict"], 
#                             name="predicted order quantity",
#                             line=dict(color='rgba(228,26,28,1)',width=2)))

#     fig.add_trace(go.Scatter(x=lstm_forecast_df["date"], 
#                               y=lstm_forecast_df["forecast"], 
#                             name="Forecasted order quantity",
#                             line=dict(color='rgba(200,20,20,1)',width=2)))

#     fig.add_trace(go.Scatter(x=holdout_wkly_df[orderdate], 
#                               y=holdout_wkly_df[orderquantity], 
#                             name="Real Order Quantity",
#                             line=dict(color='rgba(100,100,20,1)',width=2)))

# #     fig.update_layout(
# #         title="For the next week, the forecasted order quantity is {}".format(y_forecast[len(y_forecast)-1]))
    

#     return fig
    
if __name__ == "__main__":
    app.run_server(debug=True)











































