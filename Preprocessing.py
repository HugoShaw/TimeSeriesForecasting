# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:05:13 2022

@author: Hugo Xue
@email: hugo@wustl.edu; 892849924@qq.com

"""

# TODO: combine the new dataset, and clear the data set into a standard way

import pandas as pd
import numpy as np

# year 2019
data = pd.read_excel("./data/data.xlsx", parse_dates=['gii__OrderDate__c'])

# data.head()

# data.info()

data_2020_2021= pd.read_excel("../new_transaction_data2.xlsx",
                              parse_dates=['gii__OrderDate__c'])

data_2022 = pd.read_excel("./data/WashUSpring2022DataLoad_2022_04_01 mar 22.xlsx",
                          parse_dates=['gii__OrderDate__c'])

# data.tail()
# data_2020_2021.tail()

data_2022 = data_2022[data_2022['gii__OrderDate__c'] > '2022-02-28']

dt1 = data.copy()
dt2 = data_2020_2021[data_2020_2021['gii__OrderDate__c'] > '2019-12-31']

dt1['gii__OrderDate__c'].max()
dt2['gii__OrderDate__c'].max()
data_2022['gii__OrderDate__c'].max()
# total df
df = pd.concat([dt1, dt2])

df.info()

# create datatime series
import datetime

def create_assist_date(datestart = None,dateend = None):

    if datestart is None:
        datestart = '2016-01-01'
    if dateend is None:
        dateend = datetime.datetime.now().strftime('%Y-%m-%d')

    datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
    dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
    date_list = []
    date_list.append(datestart.strftime('%Y-%m-%d'))
    while datestart<dateend:
        datestart+=datetime.timedelta(days=+1)
        date_list.append(datestart.strftime('%Y-%m-%d'))
    return date_list

date_list = create_assist_date(datestart='2019-05-08', dateend='2022-02-28')

# clean data set
orderquantity = 'sum(gii__OrderQuantity__c)'
orderdate = 'gii__OrderDate__c'
warehouse = 'gii__Description__c'
productSKU = 'giic_Product_SKU__c'

DC = ['ASC- WHITTIER', 'ASC- MONROE', 'ASC- ATLANTA', 'ASC- FIFE',
    'ASC- DENVER', 'ASC- STOCKTON', 'ASC - BARTLETT', 'ASC- ORRVILLE',
    'ASC- SCHERTZ', 'ASC- ORLANDO']

# "datetime"  "order quantities" for every sku at each dc

# each warehouse
# dt = data[data[warehouse]==dc]

# which are the fast moving SKUs
new_df = df.groupby([warehouse, productSKU, orderdate]).agg({orderquantity:"sum"}).reset_index()
# new_df.info()
# new_df.head().T
datetime_list = pd.to_datetime(date_list)

tmp_list = []

for dc in DC:    
    selected_dc = new_df[new_df[warehouse] == dc]
    SKUs = pd.unique(selected_dc[productSKU])
    # print(SKUs)
    for sku in SKUs:
        created_df = pd.DataFrame(
            {warehouse: [dc for i in range(len(datetime_list))],
            productSKU: [sku for i in range(len(datetime_list))],
            orderdate: datetime_list})
        
        selected_df = selected_dc[selected_dc[productSKU] == sku]
        # for row in selected_df.iterrows():
        # merge (left join)
        result = pd.merge(created_df, selected_df, 
                          how="left", on=[warehouse, productSKU, orderdate])
        result.fillna(0, inplace=True)  
        tmp_list.append(result)

daily_df = pd.concat(tmp_list)
daily_df[orderdate].max()

daily_df.to_csv("./data/daily_transactions.csv")
        # converted to weekly data
        # result = result.resample('W-Mon',label='right',closed='right',on=orderdate)\
        #     .sum().reset_index().sort_values(by=orderdate)
        # print(result.head())
        
# .sum().reset_index().sort_values(by=orderdate)
# grp_skuDT.set_index(orderdate, inplace=True)
# # daily order quantity for each sku in each DC
# grp_skuDT = grp_skuDT.asfreq(freq='d',fill_value=0)
# # weekly order quantity for each DC
# grp_skuDT = grp_skuDT.reset_index()
# grp_skuDT = grp_skuDT.resample('W-Mon',label='right',closed='right',on=orderdate)\
# .sum().reset_index().sort_values(by=orderdate)
# grp_skuDT.set_index(orderdate, inplace=True)
# grp_skuDT = grp_skuDT.reset_index()



date_list = create_assist_date(datestart='2022-03-01', dateend='2022-03-31')

# clean data set
orderquantity = 'sum(gii__OrderQuantity__c)'
orderdate = 'gii__OrderDate__c'
warehouse = 'gii__Description__c'
productSKU = 'giic_Product_SKU__c'

DC = ['ASC- WHITTIER', 'ASC- MONROE', 'ASC- ATLANTA', 'ASC- FIFE',
    'ASC- DENVER', 'ASC- STOCKTON', 'ASC - BARTLETT', 'ASC- ORRVILLE',
    'ASC- SCHERTZ', 'ASC- ORLANDO']

# "datetime"  "order quantities" for every sku at each dc

# each warehouse
# dt = data[data[warehouse]==dc]

# which are the fast moving SKUs
new_df = data_2022.groupby([warehouse, productSKU, orderdate]).agg({orderquantity:"sum"}).reset_index()
# new_df.info()
# new_df.head().T
datetime_list = pd.to_datetime(date_list)

tmp_list = []

for dc in DC:    
    selected_dc = new_df[new_df[warehouse] == dc]
    SKUs = pd.unique(selected_dc[productSKU])
    # print(SKUs)
    for sku in SKUs:
        created_df = pd.DataFrame({warehouse: [dc for i in range(len(datetime_list))],
            productSKU: [sku for i in range(len(datetime_list))],
            orderdate: datetime_list})
        
        selected_df = selected_dc[selected_dc[productSKU] == sku]
        # for row in selected_df.iterrows():
        # merge (left join)
        result = pd.merge(created_df, selected_df, 
                          how="left", on=[warehouse, productSKU, orderdate])
        result.fillna(0, inplace=True)  
        tmp_list.append(result)

daily_df = pd.concat(tmp_list)    
daily_df[orderdate].max()

daily_df.to_csv("./data/daily_holdout.csv")



import pandas as pd
data = pd.read_csv("./data/daily_transactions.csv", index_col=0, parse_dates=[orderdate])
orderquantity = 'sum(gii__OrderQuantity__c)'
orderdate = 'gii__OrderDate__c'
warehouse = 'gii__Description__c'
productSKU = 'giic_Product_SKU__c'

data.info()
DC = ['ASC- WHITTIER', 'ASC- MONROE', 'ASC- ATLANTA', 'ASC- FIFE',
    'ASC- DENVER', 'ASC- STOCKTON', 'ASC - BARTLETT', 'ASC- ORRVILLE',
    'ASC- SCHERTZ', 'ASC- ORLANDO']
dt = data[data[warehouse] == 'ASC- ORRVILLE']
dt = dt[dt[productSKU] == "41006"]

len(dt[orderdate])
len(pd.unique(dt[orderdate]))
dt = dt.resample('W-Mon', label='right', on=orderdate)\
    .sum().reset_index().sort_values(by=orderdate)
dt.info()
from matplotlib import pyplot as plt
plt.plot(dt[orderdate], dt[orderquantity])
