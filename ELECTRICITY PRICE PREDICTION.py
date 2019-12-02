# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:47:59 2019

@author: Lenovo
"""

#Import modules and classes

# *** Start Data_Acquisition_CAISO_Prices_Load_Fcasts
%matplotlib nbagg
import wget, os, zipfile
import time
import os
print(os.getcwd())
# Functions to Construct Queries (for Accessing CAISO OASIS API)
# CSV download query format... DAM
'''http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=PRC_LMP&version=1&startdat
etime=20160101T08:00-0000&enddatetime=20160201T08:00-
0000&market_run_id=DAM&node=BAYSHOR2_1_N001

#http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=PRC_LMP&version=3&startdat
etime=20160101T08:00-0000&enddatetime=20160201T07:00-
0000&market_run_id=DAM&node=BAYSHOR2_1_N001

CSV download query format... HASP
http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=PRC_HASP_LMP&version=3&sta
rtdatetime=20160101T08:00-0000&enddatetime=20160202T08:00-
0000&market_run_id=HASP&node=BAYSHOR2_1_N001

CSV download query format... RTM
http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=PRC_INTVL_LMP&version=3&sta
rtdatetime=20190505T07:00-0000&enddatetime=20190505T08:00-
0000&market_run_id=RTM&node=BAYSHOR2_1_N001

CSV download query format... 7 Day Ahead Load F'cast
http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=SLD_FCST&version=1&market_r
un_id=7DA&startdatetime=20190505T07:00-0000&enddatetime=20190506T07:00-0000

CSV download query format... 2 Day Ahead Load F'cast
http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=SLD_FCST&version=1&market_r
un_id=2DA&startdatetime=20190505T07:00-0000&enddatetime=20190506T07:00-0000

CSV download query format... DAM Load F'cast
http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=SLD_FCST&version=1&market_r
un_id=DAM&startdatetime=20190505T07:00-0000&enddatetime=20190506T07:00-0000

CSV download query format... RTM Load F'cast
http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=SLD_FCST&version=1&market_r
un_id=RTM&startdatetime=20190505T07:00-0000&enddatetime=20190506T07:00-0000

Mandatory Parameters

Parameter Description
startdatetime valid operating start datetime in GMT (yyyymmddThh24:miZ)
enddatetime valid operating end datetime in GMT (yyyymmddThh24:miZ)
resultformat '6' for .csv file download
queryname 'PRC_LMP' for DAM LMP prices, 'PRC_HASP_LMP' for HASP LMP prices
market_run_id 'DAM' for Day-Ahead-Market, 'HASP' for Hour-Ahead-Scheduling Process
version API version ('1' for the DAM query, '3' for HASP query)
node Use the San Francisco Bay Shore node site as the example node
'''

# Function to construct a one-month electricity Day-Ahead-Market
# data query for the CAISO OASIS API
# Parameters (all strings):
# node name of node
# startyear, startmonth
def create_DAM_query(node, startyear, startmonth):
    oasis_website = 'oasis.caiso.com'
    context_path = 'oasisapi'

url = f'http://{oasis_website}/{context_path}/SingleZip'

resultformat = '6'
queryname = 'PRC_LMP'
version = '1'
market_run_id = 'DAM'

   if startmonth == '12':
       endmonth = '01'
       endyear = f'{int(startyear) + 1}'
   else:
       endmonth = f'{(int(startmonth) + 1):02d}'
       endyear = startyear

startdatetime = f'{startyear}{startmonth}01T08:00-0000'

enddatetime = f'{endyear}{endmonth}01T07:00-0000'

query = f'{url}?resultformat={resultformat}&queryname={queryname}\
&version={version}&startdatetime={startdatetime}&enddatetime={enddatetime}\
&market_run_id={market_run_id}&node={node}'

print(query)
return query

# Function to construct a one-month electricity Hour-Ahead-Scheduling-Process
# data query for the CAISO OASIS API
# Parameters (all strings):
# node name of node
# startyear, startmonth
def create_HASP_query(node, startyear, startmonth):
    oasis_website = 'oasis.caiso.com'
    context_path = 'oasisapi'

url = f'http://{oasis_website}/{context_path}/SingleZip'

resultformat = '6'
queryname = 'PRC_HASP_LMP'

version = '3'
market_run_id = 'HASP'

   if startmonth == '12':
       endmonth = '01'
       endyear = f'{int(startyear) + 1}'
   else:
       endmonth = f'{(int(startmonth) + 1):02d}'
       endyear = startyear

startdatetime = f'{startyear}{startmonth}01T08:00-0000'
enddatetime = f'{endyear}{endmonth}01T07:00-0000'

query = f'{url}?resultformat={resultformat}&queryname={queryname}\
&version={version}&startdatetime={startdatetime}&enddatetime={enddatetime}\
&market_run_id={market_run_id}&node={node}'

print(query)
return query

# Function to construct a GENERIC one-month .csv LMP price download

# data query for the CAISO OASIS API
# Parameters (all strings):
# node name of node
# startyear, startmonth
def price_query(queryname, version, startyear, startmonth, market_run_id, node):
    oasis_website = 'oasis.caiso.com'
    context_path = 'oasisapi'

url = f'http://{oasis_website}/{context_path}/SingleZip'

resultformat = '6'

    if startmonth == '12':
        endmonth = '01'
        endyear = f'{int(startyear) + 1}'
    else:
        endmonth = f'{(int(startmonth) + 1):02d}'
        endyear = startyear

startdatetime = f'{startyear}{startmonth}01T08:00-0000'
enddatetime = f'{endyear}{endmonth}01T07:00-0000'

query = f'{url}?resultformat={resultformat}&queryname={queryname}&version={version}\
&startdatetime={startdatetime}&enddatetime={enddatetime}\
&market_run_id={market_run_id}&node={node}'

print(query)
return query

# Function to construct a GENERIC one-month .csv load (demand) forecast
# download data query for the CAISO OASIS API
# Parameters (all strings):
# startyear, startmonth
def load_query(queryname, version, startyear, startmonth, market_run_id):
    oasis_website = 'oasis.caiso.com'
    context_path = 'oasisapi'

url = f'http://{oasis_website}/{context_path}/SingleZip'

resultformat = '6'

    if startmonth == '12':
        endmonth = '01'
        endyear = f'{int(startyear) + 1}'
    else:
        endmonth = f'{(int(startmonth) + 1):02d}'
        endyear = startyear

startdatetime = f'{startyear}{startmonth}01T08:00-0000'
enddatetime = f'{endyear}{endmonth}01T07:00-0000'

query = f'{url}?resultformat={resultformat}&queryname={queryname}&version={version}\
&market_run_id={market_run_id}&startdatetime={startdatetime}&enddatetime={enddatetime}'
print(query)
return query

# Get Energy Price Zipfiles (from CAISO OASIS website via API)
node = 'BAYSHOR2_1_N001'
  for i in range(1, 2): #for i in range(1, 41):
      if i % 12 == 0:
          startyear = str(2016 + i//12 - 1)
          startmonth = f'{12:02d}'
      else:
          startyear = str(2016 + i//12)
          startmonth = f'{i%12:02d}'

prc_qry = price_query('PRC_LMP',
'1',
startyear,

startmonth,
'DAM',
node)
print(prc_qry)
# Uncomment below 2 lines when running the first time to download files
wget.download(prc_qry, 'raw_data3/caiso_downloads/caiso_dam_dl/')
time.sleep(5)

# S2
for i in range(1, 2): #for i in range(1, 41):
    if i % 12 == 0: 
        startyear = str(2016 + i//12 - 1)
        startmonth = f'{12:02d}'
    else:
        startyear = str(2016 + i//12)

startmonth = f'{i%12:02d}'

prc_qry = price_query('PRC_HASP_LMP',
'3',
startyear,
startmonth,
'HASP',
node)
print(prc_qry)
# Uncomment when running the first time to download these files
wget.download(prc_qry, 'raw_data3/caiso_downloads/caiso_hasp_dl/')
time.sleep(5)
wget.download(price_query('PRC_INTVL_LMP','3',startyear,startmonth,'RTM',node),
'raw_data3/caiso_downloads/caiso_rtm_dl/')
time.sleep(5)

# S3
for i in range(1, 2): #for i in range(1, 41):
    if i % 12 == 0:
        startyear = str(2016 + i//12 - 1)
        startmonth = f'{12:02d}'
    else:
        startyear = str(2016 + i//12)

startmonth = f'{i%12:02d}'
qry1 = load_query('SLD_FCST',
'1',
startyear,
startmonth,
'7DA',
)
print(qry1)
# Uncomment when running the first time to download these files
wget.download(qry1, 'raw_data3/caiso_downloads/caiso_7da_load_dl/')
time.sleep(5)

qry2 = load_query('SLD_FCST',
'1',
startyear,
startmonth,
'2DA',
)
print(qry2)
# Uncomment this
wget.download(qry2, 'raw_data3/caiso_downloads/caiso_2da_load_dl/')
time.sleep(5)

qry3 = load_query('SLD_FCST',
'1',

startyear,
startmonth,
'DAM',
)
print(qry3)
# Uncomment when running the first time to download these files
wget.download(qry3, 'raw_data3/caiso_downloads/caiso_dam_load_dl/')
time.sleep(5)

qry4 = load_query('SLD_FCST',
'1',
startyear,
startmonth,
'RTM',
)
print(qry4)
# Uncomment when running the first time to download these files
wget.download(qry4, 'raw_data3/caiso_downloads/caiso_rtm_load_dl/')
time.sleep(5)

# S4
# Single month download cell

caiso_dam_dl = 'raw_data3/caiso_downloads/caiso_dam_dl/'
caiso_hasp_dl = 'raw_data3/caiso_downloads/caiso_hasp_dl/'

node = 'BAYSHOR2_1_N001'

startyear = '2016'
startmonth = '05'

q1 = create_DAM_query(node, startyear, startmonth)
print(q1)
# Uncomment when running the first time to download these files
wget.download(q1, caiso_dam_dl)
time.sleep(10)

# S5
q2 = create_HASP_query(node, startyear, startmonth)
print(q2)
# Uncomment when running the first time to download these files
wget.download(q2, caiso_hasp_dl)

# S6
# Unzip Downloaded CAISO Files (to .csv Directories)
unzipped_caiso_dam = 'raw_data3/unzipped_caiso/unzipped_caiso_dam/'
unzipped_caiso_hasp = 'raw_data3/unzipped_caiso/unzipped_caiso_hasp/'

caiso_dam_dl = 'raw_data3/caiso_downloads/caiso_dam_dl/'
caiso_hasp_dl = 'raw_data3/caiso_downloads/caiso_hasp_dl/'

for item in os.listdir(caiso_dam_dl): # loop through items in dir
    if item.split('.')[-1] == 'zip': # check for zip extension
        file_name = f'{caiso_dam_dl}{item}' # get relative path of files
        print(f'unzipping... {file_name}')
        zip_ref = zipfile.ZipFile(file_name) # create zipfile object

with zip_ref as target:
    target.extractall(unzipped_caiso_dam)
else: continue

# S7

   for item in os.listdir(caiso_hasp_dl): # loop through items in dir
       if item.split('.')[-1] == 'zip': # check for zip extension
           file_name = f'{caiso_hasp_dl}{item}' # get relative path of files

           print(f'unzipping... {file_name}')
          zip_ref = zipfile.ZipFile(file_name) # create zipfile object

with zip_ref as target:
    target.extractall(unzipped_caiso_hasp)
else: continue

# S8
download_dir = 'raw_data3/caiso_downloads/caiso_rtm_dl/'
unzipped_target_dir = 'raw_data3/unzipped_caiso/unzipped_caiso_rtm/'

for item in os.listdir(download_dir): # loop through items in dir
    if item.split('.')[-1] == 'zip': # check for zip extension
        file_name = f'{download_dir}{item}' # get relative path of files
        print(f'unzipping... {file_name}')
        zip_ref = zipfile.ZipFile(file_name) # create zipfile object

with zip_ref as target:
    target.extractall(unzipped_target_dir)
else: continue

# S9
def unzip_dir(download_dir, unzipped_target_dir):
    for item in os.listdir(download_dir): # loop through items in dir
        if item.split('.')[-1] == 'zip': # check for zip extension
            file_name = f'{download_dir}{item}' # get relative path of files
            print(f'unzipping... {file_name}')
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object

with zip_ref as target:
    target.extractall(unzipped_target_dir)
else: continue

# Uncomment when running the first time to unzip
unzip_dir('raw_data3/caiso_downloads/caiso_7da_load_dl/',
'raw_data3/unzipped_caiso/unzipped_caiso_7da_load/')

unzip_dir('raw_data3/caiso_downloads/caiso_2da_load_dl/',
'raw_data3/unzipped_caiso/unzipped_caiso_2da_load/')

unzip_dir('raw_data3/caiso_downloads/caiso_dam_load_dl/',
'raw_data3/unzipped_caiso/unzipped_caiso_dam_load/')

unzip_dir('raw_data3/caiso_downloads/caiso_rtm_load_dl/',
'raw_data3/unzipped_caiso/unzipped_caiso_rtm_load/')

# *** End 1a_Data_Acquisition_CAISO_Prices_Load_Fcasts

# S10
# *** Start Data_Acquisition_CA_Water_Levels

import wget, os, zipfile
import time

# Functions to Construct Queries (for Accessing CAISO OASIS API)
'''CSV download query format... hourly storage at a single reservoir
http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=SHA&SensorNums=15&dur_code=
H&Start=2016-01-01&End=2019-05-06

Excel download query format... hourly storage at a single reservoir
http://cdec.water.ca.gov/dynamicapp/req/ExcelDataServlet?Stations=SHA&SensorNums=15&dur_code
=H&Start=2016-01-01&End=2019-05-06

Mandatory Parameters
Parameter Description
placeholder ... ...'''

# Function to construct a GENERIC one-month .csv download query
# for hourly reservoir storage amounts from the CA DWR API
# Parameter (string): reservoir alphabetic code
def res_stg_query(station):
    url = f'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet'
    Stations = station
    SensorNums = '15'
    dur_code = 'H'
    Start = '2016-01-01'
    End = '2019-05-07'

query = f'{url}?Stations={station}&SensorNums={SensorNums}\
&dur_code={dur_code}&Start={Start}&End={End}'
print(query)
return query

res_stg_query('CLE')

#
'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=CLE&SensorNums=15&dur_code=
H&Start=2016-01-01&End=2019-05-07'

# S11
# Get Hourly Reservoir Storage Sensor Readings (in acre-feet from CA DWR website via API)
reservoirs_list = ['CLE', 'WHI', 'LEW', 'WRS',
'COY', 'SHA', 'KES', 'ORO',
'ANT', 'FRD', 'DAV', 'BUL',
'ENG', 'FOL', 'UNV', 'LON',
'ICH', 'NAT', 'INV', 'BER',
'BLB', 'NHG', 'CMN', 'PAR',
'DON', 'BRD', 'TUL', 'NML',
'DNP', 'HTH', 'CHV', 'EXC',
'BUC', 'HID', 'MIL', 'SNL',

'PNF', 'TRM', 'SCC', 'ISB',
'STP', 'INP', 'DNN', 'CCH',
'PYM', 'CAS', 'PRR'
]
len(reservoirs_list)
# 47

for station in reservoirs_list:
    print(res_stg_query(station))
# Uncomment when running the first time to download these files
#wget.download(res_stg_query(station), f'raw_data3/ca_dwr_dl/{station}_hrly_strg.csv')
#time.sleep(5)

# S12 skip
# Single reservoir download
# station = 'CLE'
# wget.download(res_stg_query(station), f'../ca_dwr_dl/{station}_hrly_strg.csv')

# *** End of 1b_Data_Acquisition_CA_Water_Levels

# S13

# *** Start Electricity_Prices_Dataframe

import pandas as pd
import numpy as np
import wget, os
import time
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
%matplotlib inline
sns.set_style('whitegrid')

# Day-Ahead-Market DataFrame
dam_orig_cols=['INTERVALSTARTTIME_GMT',
'OPR_DT',
'OPR_HR',
'NODE',
'MARKET_RUN_ID',
'LMP_TYPE',
'MW']

dam_new_cols =['start_datetime',
'date',

'hr_index',
'node',
'market',
'lmp_type',
'dam_price_per_mwh']

dam_rename_dict = {old: new for old, new in zip(dam_orig_cols, dam_new_cols)}
dam_rename_dict

# S14
caiso_dam_df = pd.DataFrame(columns=dam_new_cols)
caiso_dam_df.head()

# S15
for file in glob.glob('raw_data3/unzipped_caiso/unzipped_caiso_dam/*.csv'):
    df = pd.read_csv(file, usecols=dam_orig_cols).rename(index=str,
                    columns=dam_rename_dict)
df = df[df.lmp_type == 'LMP']
caiso_dam_df = caiso_dam_df.append(df, ignore_index=True)

print(caiso_dam_df.shape)
caiso_dam_df.head()
# ntc hr_index col

caiso_dam_df = caiso_dam_df.sort_values(by='start_datetime').reset_index(drop=True)

caiso_dam_df.head()
# ntc hr_index col

caiso_dam_df['start_datetime'] = pd.to_datetime(caiso_dam_df['start_datetime'])
caiso_dam_df.head()

caiso_dam_df.set_index('start_datetime', inplace=True)
caiso_dam_df.sort_index(inplace=True)
caiso_dam_df.head()

caiso_dam_df.info()

# Hour-Ahead-Scheduling Process DataFrame (hour-ahead, 15-minute realtime market)
hasp_orig_cols=['INTERVALSTARTTIME_GMT',
'OPR_DT',
'OPR_HR',
'NODE',
'MARKET_RUN_ID',
'LMP_TYPE',
'MW']
hasp_new_cols =['start_datetime',
'date',
'hr_index',
'node',
'market',
'lmp_type',
'hasp_price_per_mwh']
hasp_rename_dict = {old: new for old, new in zip(hasp_orig_cols, hasp_new_cols)}
caiso_hasp_df = pd.DataFrame(columns=hasp_new_cols)

for file in glob.glob('raw_data3/unzipped_caiso/unzipped_caiso_hasp/*.csv'):
    df = pd.read_csv(file, usecols=hasp_orig_cols).rename(index=str,
                    columns=hasp_rename_dict) 
    df = df[df.lmp_type == 'LMP']
    caiso_hasp_df = caiso_hasp_df.append(df, ignore_index=True)
    print(caiso_hasp_df.shape)  
# (116272, 7)
caiso_hasp_df = caiso_hasp_df.sort_values(by='start_datetime').reset_index(drop=True)
caiso_hasp_df['start_datetime'] = pd.to_datetime(caiso_hasp_df['start_datetime'])
caiso_hasp_df.set_index('start_datetime', inplace=True)
caiso_hasp_df.sort_index(inplace=True)
caiso_hasp_df.head()

caiso_hasp_df = caiso_hasp_df.resample('H').mean()
caiso_hasp_df.head()

print(caiso_hasp_df.shape)

# RTM DataFrame (realtime spot prices, 5-minute realtime settlements only)
rtm_orig_cols=['INTERVALSTARTTIME_GMT',
'OPR_DT',
'OPR_HR',
'NODE',
'MARKET_RUN_ID',
'LMP_TYPE',
'VALUE']
rtm_new_cols =['start_datetime',
'date',
'hr_index',
'node',
'market',
'lmp_type',
'rtm_price_per_mwh']
rtm_rename_dict = {old: new for old, new in zip(rtm_orig_cols, rtm_new_cols)}
caiso_rtm_df = pd.DataFrame(columns=rtm_new_cols)
for file in glob.glob('raw_data3/unzipped_caiso/unzipped_caiso_rtm/*.csv'):
df = pd.read_csv(file, usecols=rtm_orig_cols).rename(index=str,

columns=rtm_rename_dict)
df = df[df.lmp_type == 'LMP']
caiso_rtm_df = caiso_rtm_df.append(df, ignore_index=True)
print(caiso_rtm_df.shape)
#(349728, 7)
caiso_rtm_df = caiso_rtm_df.sort_values(by='start_datetime').reset_index(drop=True)
caiso_rtm_df['start_datetime'] = pd.to_datetime(caiso_rtm_df['start_datetime'])
caiso_rtm_df.set_index('start_datetime', inplace=True)
caiso_rtm_df.sort_index(inplace=True)
caiso_rtm_df.head()

print(caiso_rtm_df.shape)

caiso_dam_df.head()

# Join DAM + HASP + RTM LMP's Into a Single DataFrame
elec_prices_hrly = caiso_dam_df.drop(columns=['market', 'lmp_type'])
print(elec_prices_hrly.shape)
#(29149, 4)
elec_prices_hrly.head()

elec_prices_hrly = elec_prices_hrly.merge(caiso_hasp_df[['hasp_price_per_mwh']],

how='outer',
left_index = True,
right_index = True)
print(elec_prices_hrly.shape)
# (29183, 5)
elec_prices_hrly.head(15)

elec_prices_hrly = elec_prices_hrly.merge(caiso_rtm_df[['rtm_price_per_mwh']],
how='outer',
left_index = True,
right_index = True)
print(elec_prices_hrly.shape)
# (29183, 6)
elec_prices_hrly.head()

elec_prices_hrly.shape
# (29183, 6)

print(elec_prices_hrly.shape)
print(elec_prices_hrly.isna().sum())

elec_prices_hrly = elec_prices_hrly[elec_prices_hrly.hasp_price_per_mwh.notna()]
print(elec_prices_hrly.shape)
print(elec_prices_hrly.isna().sum())
#elec_prices_hrly.isna().sum()

elec_prices_hrly[elec_prices_hrly.rtm_price_per_mwh.isna()]

elec_prices_hrly.isna().sum().sum()

elec_prices_hrly.rtm_price_per_mwh.fillna(method='ffill', inplace=True)

elec_prices_hrly.isna().sum().sum()

# 0

elec_prices_hrly.head()

print(elec_prices_hrly.shape)

elec_prices_hrly.to_csv('raw_data3/data/elec_prices_hrly.csv')

plt.figure(figsize=(15,9))
plt.title('Electricty Prices (hrly): Realtime vs. Hour-ahead vs. Day-ahead', fontsize=18)

plt.plot(elec_prices_hrly.index,
elec_prices_hrly.rtm_price_per_mwh,
label = 'RTM', alpha=0.7)

plt.plot(elec_prices_hrly.index,
elec_prices_hrly.hasp_price_per_mwh,
label= 'HASP', alpha=0.9)

plt.plot(elec_prices_hrly.index,
elec_prices_hrly.dam_price_per_mwh,
label = 'DAM', alpha=0.5)

plt.ylabel('Prices in $/MWh', fontsize=18)
plt.legend()
plt.savefig('raw_data3/images/elec_prices_hrly.jpg', bbox_inches='tight')
;

# *** End of 2a_Electricity_Prices_Dataframe

'''import pandas as pd
import numpy as np

import wget, os
import time
import glob

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
%matplotlib inline
sns.set_style('whitegrid')'''

# *** Start Electricity_Demand_Fcast_Dataframe

# 7-Day-Ahead Demand Forecast DataFrame
sev_da_load_orig_cols=['INTERVALSTARTTIME_GMT',
'OPR_DT',
'OPR_HR',
'TAC_AREA_NAME',
'MW']
sev_da_load_new_cols =['start_datetime',
'date',
'hr_index',
'area',
'7da_load_fcast_mw']

sev_da_load_rename_dict = {old: new for old, new in zip(sev_da_load_orig_cols,
sev_da_load_new_cols)}
sev_da_load_rename_dict

sev_da_load_df = pd.DataFrame(columns=sev_da_load_new_cols)
sev_da_load_df.head()

for file in glob.glob('raw_data3/unzipped_caiso/unzipped_caiso_7da_load/*.csv'):
    df = pd.read_csv(file, usecols=sev_da_load_orig_cols).rename(index=str,
                    columns=sev_da_load_rename_dict)
    df = df[df.area == 'CA ISO-TAC']
    sev_da_load_df = sev_da_load_df.append(df, ignore_index=True)

    print(sev_da_load_df.shape)
    sev_da_load_df.head()

    sev_da_load_df = sev_da_load_df.sort_values(by='start_datetime').reset_index(drop=True)
    sev_da_load_df['start_datetime'] = pd.to_datetime(sev_da_load_df['start_datetime'])
    sev_da_load_df.set_index('start_datetime', inplace=True)
    sev_da_load_df.sort_index(inplace=True)
    sev_da_load_df.head()

    sev_da_load_df.info()

# 2-Day-Ahead Demand Forecast DataFrame
two_da_load_orig_cols=['INTERVALSTARTTIME_GMT',
                       'OPR_DT',
                       'OPR_HR',
                       'TAC_AREA_NAME','MW']
two_da_load_new_cols =['start_datetime',
                       'date',
                       'hr_index',
                       'area',
                       '2da_load_fcast_mw']
two_da_load_rename_dict = {old: new for old, new in zip(two_da_load_orig_cols,
                                                        two_da_load_new_cols)}
    two_da_load_df = pd.DataFrame(columns=two_da_load_new_cols)

for file in glob.glob('raw_data3/unzipped_caiso/unzipped_caiso_2da_load/*.csv'):
    df = pd.read_csv(file, usecols=two_da_load_orig_cols).rename(index=str,
                    columns=two_da_load_rename_dict)
    df = df[df.area == 'CA ISO-TAC']
    two_da_load_df = two_da_load_df.append(df, ignore_index=True)

    two_da_load_df.shape
#(29144, 5)
    two_da_load_df = two_da_load_df.sort_values(by='start_datetime').reset_index(drop=True)
    two_da_load_df['start_datetime'] = pd.to_datetime(two_da_load_df['start_datetime'])
    two_da_load_df.set_index('start_datetime', inplace=True)
    two_da_load_df.sort_index(inplace=True)
    two_da_load_df.head()

    two_da_load_df.info()

# Day-Ahead Demand Forecast DataFrame
dam_load_orig_cols=['INTERVALSTARTTIME_GMT',
                    'OPR_DT',
                    'OPR_HR',
                    'TAC_AREA_NAME','MW']
dam_load_new_cols =['start_datetime',
                    'date',
                    'hr_index',
                    'area',
                    'dam_load_fcast_mw']
#dam_load_rename_dict = {old: new for old, new in zip(dam_load_orig_cols,
dam_load_rename_dict = {old: new for old, new in zip(dam_load_orig_cols,
                                                     dam_load_new_cols)}
dam_load_df = pd.DataFrame(columns=dam_load_new_cols)
for file in glob.glob('raw_data3/unzipped_caiso/unzipped_caiso_dam_load/*.csv'):
    df = pd.read_csv(file, usecols=dam_load_orig_cols).rename(index=str,
                    columns=dam_load_rename_dict)
    df = df[df.area == 'CA ISO-TAC']
    dam_load_df = dam_load_df.append(df, ignore_index=True)
    dam_load_df.shape
# (29144, 5)

    dam_load_df = dam_load_df.sort_values(by='start_datetime').reset_index(drop=True)
    dam_load_df['start_datetime'] = pd.to_datetime(dam_load_df['start_datetime'])
    dam_load_df.set_index('start_datetime', inplace=True)
    dam_load_df.sort_index(inplace=True)
    dam_load_df.head()

    dam_load_df.info()

# RTM DataFrame (realtime spot prices, 5-minute realtime settlements only)
rtm_load_orig_cols=['INTERVALSTARTTIME_GMT',
                    'OPR_DT',
                    'OPR_HR',
                    'TAC_AREA_NAME',
                    'MW']

rtm_load_new_cols =['start_datetime',
                    'date',
                    'hr_index',
                    'area',
                    'rtm_load_fcast_mw']

rtm_load_rename_dict = {old: new for old, new in zip(rtm_load_orig_cols,
                                                     rtm_load_new_cols)}
    rtm_load_df = pd.DataFrame(columns=rtm_load_new_cols)

for file in glob.glob('raw_data3/unzipped_caiso/unzipped_caiso_rtm_load/*.csv'):
    df = pd.read_csv(file, usecols=rtm_load_orig_cols).rename(index=str,
                    columns=rtm_load_rename_dict)
    df = df[df.area == 'CA ISO-TAC']
    rtm_load_df = rtm_load_df.append(df, ignore_index=True)

    rtm_load_df.shape
# (464532, 5)

    rtm_load_df = rtm_load_df.sort_values(by='start_datetime').reset_index(drop=True)
    rtm_load_df['start_datetime'] = pd.to_datetime(rtm_load_df['start_datetime'])

    rtm_load_df.set_index('start_datetime', inplace=True)
    rtm_load_df.sort_index(inplace=True)
    rtm_load_df.head()

    rtm_load_df = rtm_load_df.resample('H').mean()

    rtm_load_df.head()

    rtm_load_df.shape
#(29183, 1)

# Join Forecasts Into a Single DataFrame
elec_demand_hrly = sev_da_load_df.drop(columns=['area'])

print(elec_demand_hrly.shape)

#(29072, 3)

elec_demand_hrly = elec_demand_hrly.merge(two_da_load_df[['2da_load_fcast_mw']],
                                          how='outer',left_index = True,right_index = True)
    print(elec_demand_hrly.shape)
#(29144, 4)

elec_demand_hrly = elec_demand_hrly.merge(dam_load_df[['dam_load_fcast_mw']],
                                          how='outer',left_index = True,right_index = True)
    print(elec_demand_hrly.shape)
#(29144, 5)

elec_demand_hrly = elec_demand_hrly.merge(rtm_load_df[['rtm_load_fcast_mw']],
                                          how='outer',left_index = True,right_index = True)
    print(elec_demand_hrly.shape)
# (29183, 6)

elec_demand_hrly.head(10)

elec_demand_hrly.isna().sum()

elec_demand_hrly = elec_demand_hrly.dropna()
elec_demand_hrly.shape
#(29068, 6)

elec_demand_hrly.isna().sum()

elec_demand_hrly.to_csv('raw_data3/data/elec_demand_hrly.csv')

plt.figure(figsize=(15,9))
plt.title('Electricty Demand (hrly): Day-ahead', fontsize=18)
# plt.plot(elec_demand_hrly.index,
# elec_demand_hrly['7da_load_fcast_mw'],
# label = '7 DA', alpha=0.5)
# plt.plot(elec_demand_hrly.index,
# elec_demand_hrly['2da_load_fcast_mw'],
# label= '2 DA', alpha=0.2)

plt.plot(elec_demand_hrly.index,
         elec_demand_hrly['dam_load_fcast_mw'],
         label = '1 DA', alpha=0.5)
# plt.plot(elec_demand_hrly.index,
# elec_demand_hrly['rtm_load_fcast_mw'],
# label = 'Realtime', alpha=0.2)
plt.ylabel('Demand in MW', fontsize=18)
plt.legend()
# plt.savefig('raw_data3/images/elec_demand_hrly.jpg', bbox_inches='tight')
plt.show();

# *** End of 2b_Electricity_Demand_Fcast_Dataframe

# **** Start Water_Levels_Dataframe

# Reservoir Storage DataFrame Columns
res_strg_orig_cols=['DATE TIME','STATION_ID','VALUE']
res_strg_new_cols =['datetime','reservoir_id','water_acre_feet']
res_strg_rename_dict = {old: new for old, new in zip(res_strg_orig_cols,
                                                     res_strg_new_cols)}
# Loop Through Files & Append Data
res_strg_df = pd.DataFrame(columns=res_strg_new_cols)
for file in glob.glob('raw_data3/ca_dwr_dl/*.csv'):
    df = pd.read_csv(file, usecols=res_strg_orig_cols).rename(index=str,
                    columns=res_strg_rename_dict)
    res_strg_df = res_strg_df.append(df, ignore_index=True, sort=True)
    res_strg_df = res_strg_df.sort_values(by='datetime').reset_index(drop=True)
    res_strg_df['datetime'] = pd.to_datetime(res_strg_df['datetime'])
    res_strg_df.set_index('datetime', inplace=True)
    res_strg_df.sort_index(inplace=True)

    res_strg_df.head()

    res_strg_df = res_strg_df.tz_localize('America/Los_Angeles',ambiguous=False)

    res_strg_df.tail()

    print(res_strg_df.shape)
# (1080658, 2)
    res_strg_df.isna().sum()
    
    res_strg_df.replace('---', np.nan, inplace=True)
    res_strg_df.replace('ART', np.nan, inplace=True)
    res_strg_df.replace('BRT', np.nan, inplace=True)

    res_strg_df.isna().sum()

    res_strg_df.water_acre_feet.fillna(method='ffill', inplace=True)

    res_strg_df.isna().sum()

    res_strg_df.dtypes

    res_strg_df.water_acre_feet = res_strg_df.water_acre_feet.astype(int)
    res_strg_df.dtypes

    ca_water_levels = pd.DataFrame(res_strg_df.groupby('datetime')['water_acre_feet'].sum())
    ca_water_levels.head()

    ca_water_levels.shape
# (29325, 1)

    ca_water_levels.tail()
# Export to .csv File

    res_strg_df.to_csv('raw_data3/data/water_levels_by_reservoir.csv')

    ca_water_levels.to_csv('raw_data3/data/ca_water_levels.csv')

# Plot Water Levels Over Time
plt.figure(figsize=(15,9))
plt.title('Water Level Measurements (hrly)', fontsize=18)

plt.plot(ca_water_levels.index,ca_water_levels.water_acre_feet,label = 'Water Levels',alpha = 0.9)
plt.xlim('2018-10-21', '2018-10-28')

# plt.ylabel('Water Level (in Acre-feet)', fontsize=18)
plt.legend()
# plt.savefig('raw_data3/images/ca_wtr_lev_1wk_oct18.jpg', bbox_inches='tight')
plt.show();

# **** End of 2c_Water_Levels_Dataframe

# *** Start of NOAA_Four_CA_Weather_Stn_Dataframes

# Declare Variables, Directories, and Column Names
stations_dict = {'72290023188': 'sand', # San Diego
                 '72286903171': 'rive', # Riverside
                 '72592024257': 'redd', # Redding
                 '72389093193': 'fres' # Fresno
                 }
    stn_id_list = list(stations_dict.keys())
    stn_name_list = list(stations_dict.values())
    print(f'NOAA Weather Station IDs: {stn_id_list}')
    print(f'NOAA Weather Station Names: {stn_name_list}')

# Dataframe names
sand_df = None
rive_df = None
redd_df = None
fres_df = None
#NOAA Weather Station IDs: ['72290023188', '72286903171', '72592024257', '72389093193']
#NOAA Weather Station Names: ['sand', 'rive', 'redd', 'fres']

download_dirs = ['raw_data3/noaa_weather/2016/',
                 'raw_data3/noaa_weather/2017/',
                 'raw_data3/noaa_weather/2018/',
                 'raw_data3/noaa_weather/2019/'
                 ]
weather_orig_cols=['STATION',
                   'DATE',
                   'TMP',
                   'WND',
                   'CIG',
                   'VIS']
weather_new_cols =['stn_id',
                   'datetime',
                   'temp',
                   'wind',
                   'ceil',
                   'vis']
weather_rename_dict = {old: new for old, new in zip(weather_orig_cols,
                                                    weather_new_cols)}
# Define Functions
# Function to create NOAA weather station DataFrame
# Parameters: dataframe (df), index slice position in stations_list
# Function steps:
# Instantiate DataFrame
# Loop through the 4 annual download directories

# Reading in data from .csv files, appending rows to new dataframe
# Sort by datetime column, cast to 'datetime' object, and set as index
# Sort by new datetime index to show rows in chronological order

#Function to extract relevant statistics from NOAA column
# Parameters: dataframe (df), orig_col_name (string), new_col_name (string), slice_index
# Function steps:
# Step through rows of a dataframe
# Extract a single statistic from a "cell" that is a tuple of strings
# Store that stat in a new column whose name is a parameter
# Drop the original data column from the dataframe
# Function to print relevant descriptive info on dataframe
# Parameter: dataframe (df)

# Function steps:
# print df.shape
# print null counts
# print head(5)

# Function to replace hard-coded "99xxx9" values w/ NaNs
# Parameter: dataframe (df)

# Function steps:
# replace specific codes (depends on which column) NOAA uses as NaN
# return null counts

def create_stn_df(df, stn_idx):
    df = pd.DataFrame(columns=weather_new_cols)
    stn_name = list(stations_dict.values())[stn_idx]
    stn_id = list(stations_dict.keys())[stn_idx]

for directory in download_dirs:
    file = directory + stn_id + '.csv'
    temp_df = pd.read_csv(file,
                          usecols=weather_orig_cols).rename(index=str,
                                                   columns=weather_rename_dict)
    df = df.append(temp_df, ignore_index=True, sort=True)

    df = df.sort_values(by='datetime').reset_index(drop=True)

    df['datetime'] = pd.to_datetime(df['datetime'])

    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    print(f'{stn_name}_df DataFrame Created')

return df

def extract_statistic(df, orig_col_name, new_col_name, slice_index):
    for row in range(0, df.shape[0]):
        try:
            statistic = df.loc[df.index[row], orig_col_name].split(',')[slice_index]
            df.loc[df.index[row], new_col_name] = statistic
        except IndexError: break
except:
    df.drop(labels=df.index[row], axis=0, inplace=True)

    df.drop(columns=[orig_col_name], inplace=True)

return

def summarize(df):
    print(f'\nShape: {df.shape}')
    print(f'\n Nulls:\n{df.isna().sum()}')
    print(f'\ndf.head()\n{df.head()}')

return

def replace_nines(df):
    df.replace('9999', np.nan, inplace=True)
    df.replace('99999', np.nan, inplace=True)
    df.replace('999999', np.nan, inplace=True)
    df.isna().sum()

return

# Read in Data and Create 4 Weather Station DataFrames

stn_name_list
# ['sand', 'rive', 'redd', 'fres']

# Skip: These steps when we run them v get err due
# as v r unable to download these files
# so v can use the files in the raw_data dir
# in the next step
sand_df = create_stn_df(sand_df, 0)
    '''FileNotFoundError: [Errno 2] File b'raw_data3/noaa_weather/2016/72290023188.csv'
    does not exist: b'raw_data3/noaa_weather/2016/72290023188.csv'''
rive_df = create_stn_df(rive_df, 1)
redd_df = create_stn_df(redd_df, 2)
fres_df = create_stn_df(fres_df, 3)

'''sand_df DataFrame Created
rive_df DataFrame Created
redd_df DataFrame Created
fres_df DataFrame Created
# San Diego DataFrame

summarize(sand_df)
df.head()
sand_df.drop_duplicates().shape
sand_df = sand_df.drop_duplicates()
sand_df.shape
extract_statistic(sand_df, 'temp', 'sand_temp', 0)
extract_statistic(sand_df, 'wind', 'sand_wind', 3)
extract_statistic(sand_df, 'vis', 'sand_vis', 0)
extract_statistic(sand_df, 'ceil', 'sand_ceil', 0)
sand_df.drop(columns=['stn_id'], inplace=True)
summarize(sand_df)
df.head()
sand_df.dropna(inplace=True)
replace_nines(sand_df)
sand_df.isna().sum()

# decided not to fill, because they are at "odd" times
# that don't align with other measurent times (:51 past each hour)
# sand_df.sand_ceil.fillna(method='ffill', inplace=True)
# sand_df.isna().sum()
sand_df.dropna(inplace=True)
summarize(sand_df)
df.head()
sand_df.to_csv('raw_data3/data/san_diego_weather2.csv')
'''

sand_df = pd.read_csv('raw_data3/data/san_diego_weather2.csv')
sand_df.head()

summarize(sand_df)

# Riverside DataFrame

'''summarize(rive_df)
df.head()
rive_df.drop_duplicates().shape
rive_df = rive_df.drop_duplicates()
rive_df.shape
extract_statistic(rive_df, 'temp', 'rive_temp', 0)
extract_statistic(rive_df, 'wind', 'rive_wind', 3)
extract_statistic(rive_df, 'vis', 'rive_vis', 0)
extract_statistic(rive_df, 'ceil', 'rive_ceil', 0)
rive_df.drop(columns=['stn_id'], inplace=True)
summarize(rive_df)
df.head()
rive_df.dropna(inplace=True)
replace_nines(rive_df)
rive_df.isna().sum()
# did not fill forward... small num of nulls
# rive_df.sand_COLUMN.fillna(method='ffill', inplace=True)

# rive_df.isna().sum()
rive_df.dropna(inplace=True)
summarize(rive_df)

df.head()
rive_df.to_csv('raw_data3/data/riverside_weather2.csv')
'''

rive_df = pd.read_csv('raw_data3/data/riverside_weather2.csv')
rive_df.head()

summarize(rive_df)

# Redding DataFrame
'''summarize(redd_df)
df.head()
redd_df.drop_duplicates().shape
redd_df = redd_df.drop_duplicates()
redd_df.shape

extract_statistic(redd_df, 'temp', 'redd_temp', 0)
extract_statistic(redd_df, 'wind', 'redd_wind', 3)
extract_statistic(redd_df, 'vis', 'redd_vis', 0)
extract_statistic(redd_df, 'ceil', 'redd_ceil', 0)
redd_df.drop(columns=['stn_id'], inplace=True)

summarize(redd_df)
df.head()
redd_df.dropna(inplace=True)
replace_nines(redd_df)
redd_df.isna().sum()
# decided not to fill, because they are at "odd" times
# that don't align with other measurent times (:53 past each hour)
# redd_df.sand_ceil.fillna(method='ffill', inplace=True)
# redd_df.isna().sum()
redd_df.dropna(inplace=True)
summarize(redd_df)
df.head()
redd_df.to_csv('raw_data3/data/redding_weather2.csv')
'''

redd_df = pd.read_csv('raw_data3/data/redding_weather2.csv')
summarize(redd_df)

# Fresno DataFrame

'''summarize(fres_df)

df.head()

fres_df.drop_duplicates().shape

fres_df = fres_df.drop_duplicates()
fres_df.shape

extract_statistic(fres_df, 'temp', 'fres_temp', 0)
extract_statistic(fres_df, 'wind', 'fres_wind', 3)
extract_statistic(fres_df, 'vis', 'fres_vis', 0)
extract_statistic(fres_df, 'ceil', 'fres_ceil', 0)

fres_df.drop(columns=['stn_id'], inplace=True)

summarize(fres_df)

df.head()

fres_df.dropna(inplace=True)

replace_nines(fres_df)

fres_df.isna().sum()

# decided not to fill, because they are at "odd" times
# that don't align with other measurent times (:53 past each hour)

# fres_df.sand_COLUMN.fillna(method='ffill', inplace=True)
# fres_df.isna().sum()

fres_df.dropna(inplace=True)

summarize(fres_df)
# Shape: (21711, 4)
df.head()

fres_df.to_csv('raw_data3/data/fresno_weather2.csv')

'''

fres_df = pd.read_csv('raw_data3/data/fresno_weather2.csv')
summarize(fres_df)

# *** End of 2d_NOAA_Four_CA_Weather_Stn_Dataframes

# *** Start Electricity_Prices_(5min)_Dataframe

# Day-Ahead-Market DataFrame
dam_orig_cols=['INTERVALSTARTTIME_GMT',
               'OPR_DT',
               'OPR_HR',
               'NODE',
               'MARKET_RUN_ID',
               'LMP_TYPE',
               'MW']

dam_new_cols =['start_datetime',
               'date',
               'hr_index',
               'node',
               'market',
               'lmp_type',
               'dam_price_per_mwh']

dam_rename_dict = {old: new for old, new in zip(dam_orig_cols, dam_new_cols)}

caiso_dam_df = pd.DataFrame(columns=dam_new_cols)

for file in glob.glob('raw_data3/unzipped_caiso/unzipped_caiso_dam/*.csv'):
    df = pd.read_csv(file, usecols=dam_orig_cols).rename(index=str,
                    columns=dam_rename_dict)
    df = df[df.lmp_type == 'LMP']
    caiso_dam_df = caiso_dam_df.append(df, ignore_index=True)

    caiso_dam_df.shape
# (29149, 7)

    caiso_dam_df = caiso_dam_df.sort_values(by='start_datetime').reset_index(drop=True)
    
    caiso_dam_df['start_datetime'] = pd.to_datetime(caiso_dam_df['start_datetime'])

    caiso_dam_df.set_index('start_datetime', inplace=True)
    caiso_dam_df.sort_index(inplace=True)
    caiso_dam_df.head()

    caiso_dam_df.info()

# Hour-Ahead-Scheduling Process DataFrame (hour-ahead, 15-minute realtime market)
hasp_orig_cols=['INTERVALSTARTTIME_GMT',
                'OPR_DT',
                'OPR_HR',
                'NODE',
                'MARKET_RUN_ID',
                'LMP_TYPE',
                'MW']

hasp_new_cols =['start_datetime',
                'date',
                'hr_index',
                'node',
                'market',
                'lmp_type',
                'hasp_price_per_mwh']

hasp_rename_dict = {old: new for old, new in zip(hasp_orig_cols, hasp_new_cols)}

caiso_hasp_df = pd.DataFrame(columns=hasp_new_cols)

for file in glob.glob('raw_data3/unzipped_caiso/unzipped_caiso_hasp/*.csv'):
    df = pd.read_csv(file, usecols=hasp_orig_cols).rename(index=str,
                    columns=hasp_rename_dict)
    df = df[df.lmp_type == 'LMP']
    caiso_hasp_df = caiso_hasp_df.append(df, ignore_index=True)

    caiso_hasp_df.shape
# (116272, 7)

    caiso_hasp_df = caiso_hasp_df.sort_values(by='start_datetime').reset_index(drop=True)

    caiso_hasp_df['start_datetime'] = pd.to_datetime(caiso_hasp_df['start_datetime'])

    caiso_hasp_df.set_index('start_datetime', inplace=True)
    caiso_hasp_df.sort_index(inplace=True)
    caiso_hasp_df.head()

    caiso_hasp_df.info()

# RTM DataFrame (realtime spot prices, 5-minute realtime settlements only)
rtm_orig_cols=['INTERVALSTARTTIME_GMT',
               'OPR_DT',
               'OPR_HR',
               'NODE',
               'MARKET_RUN_ID',
               'LMP_TYPE',
               'VALUE']

rtm_new_cols =['start_datetime',
               'date',
               'hr_index',
               'node',
               'market',
               'lmp_type',
               'rtm_price_per_mwh']

rtm_rename_dict = {old: new for old, new in zip(rtm_orig_cols, rtm_new_cols)}

caiso_rtm_df = pd.DataFrame(columns=rtm_new_cols)

for file in glob.glob('raw_data3/unzipped_caiso/unzipped_caiso_rtm/*.csv'):
    df = pd.read_csv(file, usecols=rtm_orig_cols).rename(index=str,
                    columns=rtm_rename_dict)
    df = df[df.lmp_type == 'LMP']
    caiso_rtm_df = caiso_rtm_df.append(df, ignore_index=True)

    caiso_rtm_df.shape
# (349728, 7)

    caiso_rtm_df = caiso_rtm_df.sort_values(by='start_datetime').reset_index(drop=True)

    caiso_rtm_df['start_datetime'] = pd.to_datetime(caiso_rtm_df['start_datetime'])

    caiso_rtm_df.set_index('start_datetime', inplace=True)
    caiso_rtm_df.sort_index(inplace=True)
    caiso_rtm_df.head()

    caiso_rtm_df.info()

# Join DAM + HASP + RTM LMP's Into a Single DataFrame
elec_prices_5min = caiso_rtm_df.drop(columns=['market', 'lmp_type'])
elec_prices_5min.shape
# (349728, 4)

elec_prices_5min.head()

elec_prices_5min = elec_prices_5min.merge(caiso_hasp_df[['hasp_price_per_mwh']],
                                          how='outer',left_index = True,right_index = True)
    elec_prices_5min.shape
#(349748, 5)
    
    elec_prices_5min.head() 

    elec_prices_5min = elec_prices_5min.merge(caiso_dam_df[['dam_price_per_mwh']],
                                              how='outer',left_index = True,right_index = True)

    elec_prices_5min.shape
#(349748, 6)

    elec_prices_5min.head(15)

    elec_prices_5min.isna().sum().sum()
# 554155

    elec_prices_5min.fillna(method='ffill', inplace=True)
    elec_prices_5min.isna().sum().sum()

    print(elec_prices_5min.shape)

    elec_prices_5min.head()

    elec_prices_5min.tail()

    elec_prices_5min.to_csv('raw_data3/data/elec_prices_5min.csv')

plt.figure(figsize=(15,9))
plt.title('Electricty Prices (5 min): Realtime vs. Hour-ahead vs. Day-ahead', fontsize=18)

plt.plot(elec_prices_5min.index,
elec_prices_5min.rtm_price_per_mwh,

label = 'RTM', alpha=0.7)

plt.plot(elec_prices_5min.index,
elec_prices_5min.hasp_price_per_mwh,
label= 'HASP', alpha=0.9)

plt.plot(elec_prices_5min.index,
elec_prices_5min.dam_price_per_mwh,
label = 'DAM', alpha=0.5)

plt.ylabel('Prices in $/MWh', fontsize=18)
plt.legend()
plt.show()
plt.savefig('raw_data3/images/elec_prices_5min.jpg', bbox_inches='tight')
;

# *** End of 2x_Electricity_Prices_(5min)_Dataframe

# *** Start NOAA_Combined_CA_Weather_Dataframe-(4_stn)

def index_to_datetime(df):
    df['datetime'] = pd.to_datetime(df['datetime']).dt.round('H')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

return df

def shapes_nulls():
    print(f'sand: {sand_df.shape[0]} rows, {sand_df.isna().sum().sum()} nulls')
    print(f'rive: {rive_df.shape[0]} rows, {rive_df.isna().sum().sum()} nulls')
    print(f'redd: {redd_df.shape[0]} rows, {redd_df.isna().sum().sum()} nulls')
    print(f'fres: {fres_df.shape[0]} rows, {fres_df.isna().sum().sum()} nulls')
return
# Read in Four Individual Weather Station DataFrames

'''sand_df = pd.read_csv('raw_data3/data/san_diego_weather2.csv')
rive_df = pd.read_csv('raw_data3/data/riverside_weather2.csv')
redd_df = pd.read_csv('raw_data3/data/redding_weather2.csv')
fres_df = pd.read_csv('raw_data3/data/fresno_weather2.csv')
'''
shapes_nulls()

'''sand: 24007 rows, 0 nulls
rive: 17142 rows, 0 nulls
redd: 22201 rows, 0 nulls
fres: 21711 rows, 0 nulls'''

sand_df = index_to_datetime(sand_df)
rive_df = index_to_datetime(rive_df)

redd_df = index_to_datetime(redd_df)
fres_df = index_to_datetime(fres_df)

sand_df.head()
'''
rive_df.head()
redd_df.head()
fres_df.head()'''

# Join DataFrames

weather_df = sand_df
weather_df.shape
# (24007, 4)

weather_df.head()

weather_df = weather_df.merge(fres_df,how='outer',left_index = True,right_index = True)

weather_df.head()

weather_df.shape

#(33488, 8)

weather_df = weather_df.merge(rive_df,how='outer',left_index = True,right_index = True)weather_df.shape
# (39859, 12)

weather_df = weather_df.merge(redd_df,how='outer',left_index = True,right_index = True)weather_df.head()

# (51727, 16)

weather_df.drop_duplicates(inplace=True)

print(weather_df.shape)
# (49932, 16)
weather_df.head()

weather_df = weather_df[~weather_df.index.duplicated(keep='first')]

print(weather_df.shape)
weather_df.head()
# (26190, 16)

datetime_index = pd.date_range(start = '2016-01-01 01:00',end = '2019-04-24 07:00',freq = 'H')
len(datetime_index)

weather_df.head()

weather_df = weather_df.reindex(datetime_index)

weather_df.head()

weather_df.fillna(method='ffill', inplace=True)

weather_df = weather_df.tz_localize('America/Los_Angeles',ambiguous=True,nonexistent='shift_forward')

weather_df.isna().sum().sum()
#0

print(weather_df.info())

weather_df = weather_df.astype(int)

weather_df.info()

weather_df.shape
#(29023, 16)

weather_df.head()

weather_df.to_csv('raw_data3/data/ca_weather.csv')

# *** End of 3a_NOAA_Combined_CA_Weather_Dataframe-(4_stn)

# *** Start Fully_Consolidated_Dataframe

import wget, os
import time
import glob
import pytz
import pickle

def index_to_datetime(df):
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.round('H')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
return df

def shapes_nulls():
    print(f'Elec Prices: {elec_prices_df.shape[0]} rows, {elec_prices_df.isna().sum().sum()} nulls')
    print(f'Elec Dem Fcasts: {load_fcasts_df.shape[0]} rows, {load_fcasts_df.isna().sum().sum()} nulls')
    print(f'CA Water Levels: {water_levels_df.shape[0]} rows, {water_levels_df.isna().sum().sum()} nulls')

    print(f'CA Weather Data: {weather_df.shape[0]} rows, {weather_df.isna().sum().sum()} nulls')
return

# Read in Four Sub- DataFrames
elec_prices_df = pd.read_csv('raw_data3/data/elec_prices_hrly.csv')
load_fcasts_df = pd.read_csv('raw_data3/data/elec_demand_hrly.csv')
water_levels_df = pd.read_csv('raw_data3/data/ca_water_levels.csv')
weather_df = pd.read_csv('raw_data3/data/ca_weather.csv')

shapes_nulls()

# Set Datetime Indices

# pytz.all_timezones

elec_prices_df.rename(columns={'start_datetime': 'datetime'}, inplace=True)
elec_prices_df = index_to_datetime(elec_prices_df)
elec_prices_df = elec_prices_df.tz_convert('America/Los_Angeles')

load_fcasts_df.rename(columns={'start_datetime': 'datetime'}, inplace=True)
load_fcasts_df = index_to_datetime(load_fcasts_df)

load_fcasts_df = load_fcasts_df.tz_convert('America/Los_Angeles')

water_levels_df = index_to_datetime(water_levels_df)
water_levels_df = water_levels_df.tz_convert('America/Los_Angeles')

weather_df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
weather_df = index_to_datetime(weather_df)
weather_df = weather_df.tz_convert('America/Los_Angeles')

elec_prices_df.head()

load_fcasts_df.head()

water_levels_df.head()

weather_df.head()

# Join DataFrames

df = elec_prices_df
df.shape

#(29068, 6)

df = df.merge(load_fcasts_df[['7da_load_fcast_mw',
                              '2da_load_fcast_mw',
                              'dam_load_fcast_mw',
                              'rtm_load_fcast_mw']],how='left',left_index = True,right_index = True)
df.shape
#(29068, 10)

df = df.merge(water_levels_df,how='left',left_index = True,right_index = True)
df.shape

df = df.merge(weather_df,how='left',left_index = True,right_index = True)
df.shape
#(29072, 27)

df.drop_duplicates(inplace=True)
df.shape
#(29072, 27)

df = df[~df.index.duplicated(keep='first')]
df.shape
#(29068, 27)

df.head()

print(df.shape)
#(29067, 27)

df.head()

# *** End of 3b_Fully_Consolidated_Dataframe

# **** Start of Pre-processing_for_Continuous_Targets

def summarize(df):

print(f'{df.shape[0]} rows, {df.isna().sum().sum()} nulls')
print(f'\n {df.head(3)}')
return

# Read in Consolidated Data & Examine
'''with open('raw_data3/data/consolidated_data.pkl', 'rb') as f:
df = pickle.load(f)'''
df.head()

summarize(df)

# Add year, month, day, and hour Columns
# (drop the node, date, and hr columns
df.drop(columns=['node', 'date', 'hr_index'], inplace=True)
df.head(3)

df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day

df['hour'] = df.index.hour

df.head(3)

df.shape
#(29067, 28)

df.to_csv('raw_data3/data/pre_processed_data.csv')

with open('raw_data3/data/pre_processed_df.pkl', 'wb') as f:
pickle.dump(df, f)

# Train-test-split and Scale the Data
train_set_length = int(round((df.shape[0] * 0.75), 0))

test_set_length = df.shape[0] - train_set_length

print(f'\nTrain: {train_set_length} rows\nTest: {test_set_length} rows')

train = df.iloc[: train_set_length]
test = df.iloc[train_set_length: ]

print(f'\nTrain set shape: {train.shape}\nTest set shape: {test.shape}\n')

# Train set shape: (21800, 28)
# Test set shape: (7267, 28)

train.head(3)

test.head(3)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_sc = ss.fit_transform(train)
test_sc = ss.transform(test)

print(train_sc.shape)
# (21800, 28)
print(test_sc.shape)
#(7267, 28)

train_sc_df = pd.DataFrame(train_sc, columns=train.columns, index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=test.columns, index=test.index)
print(train_sc_df.shape)
#(21800, 28)
print(test_sc_df.shape)
#(7267, 28)
train_sc_df.head(3)

train.to_csv('raw_data3/data/processed/train.csv')
test.to_csv('raw_data3/data/processed/test.csv')

with open('raw_data3/data/processed/train.pkl', 'wb') as f:
pickle.dump(train, f)

with open('raw_data3/data/processed/test.pkl', 'wb') as f:
pickle.dump(test, f)

with open('raw_data3/data/processed/train_sc_df.pkl', 'wb') as f:
pickle.dump(train_sc_df, f)

with open('raw_data3/data/processed/test_sc_df.pkl', 'wb') as f:
pickle.dump(test_sc_df, f)

# **** End of 4_Pre-processing_for_Continuous_Targets

# *** Start of EDA and Inspection

from scipy.stats import normaltest

import pickle
import time
import glob
import pytz
from datetime import timedelta

# 2 x N subplots ...

def gen_linecharts(dataframe, list_of_columns, cols, file=None):
    rows = math.ceil(len(list_of_columns)/cols)
    figwidth = 5 * cols
    figheight = 4 * rows

    fig, ax = plt.subplots(nrows = rows,
                           ncols = cols,
                           figsize = (figwidth, figheight))
        color_choices = ['blue', 'grey', 'goldenrod', 'r', 'black', 'darkorange', 'g']

plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax = ax.ravel() # Ravel turns a matrix into a vector... easier to iterate

plt.subplots_adjust(bottom=0.00, top=1.00)

for i, column in enumerate(list_of_columns):
    ax[i].plot(dataframe[column],
      color=color_choices[i % len(color_choices)])

    ax[i].set_title(f'{column}', fontsize=18)
    ax[i].set_ylabel(f'{column}', fontsize=14)
    ax[i].set_xlabel('Time', fontsize=14)
    if file:
        plt.savefig(file, bbox_inches='tight')

    plt.show();
return

# 2 x N subplots with a user-specified rolling average...

def gen_linecharts_rolling(dataframe, roll_num, list_of_columns, cols, file=None):
    rows = math.ceil(len(list_of_columns)/cols)
    figwidth = 5 * cols
    figheight = 4 * rows

    dataframe = dataframe.rolling(roll_num).mean()

    fig, ax = plt.subplots(nrows = rows,
                           ncols = cols,
                           figsize = (figwidth, figheight))

        color_choices = ['blue', 'grey', 'goldenrod', 'r', 'black', 'darkorange', 'g']

plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax = ax.ravel() # Ravel turns a matrix into a vector... easier to iterate

plt.subplots_adjust(bottom=0.00, top=1.00)

for i, column in enumerate(list_of_columns):
    ax[i].plot(dataframe[column],

    color=color_choices[i % len(color_choices)])

    ax[i].set_title(f'{column}', fontsize=18)
    ax[i].set_ylabel(f'{column}', fontsize=14)
    ax[i].set_xlabel('Time', fontsize=14)
if file:
    plt.savefig(file, bbox_inches='tight')
    plt.show();
return

# 2 x N subplots ...

def gen_scatterplots(dataframe, target_column, list_of_columns, cols, file=None):
    rows = math.ceil(len(list_of_columns)/cols)
    figwidth = 5 * cols
    figheight = 4 * rows

    fig, ax = plt.subplots(nrows = rows,
                           ncols = cols,
                           figsize = (figwidth, figheight))

        color_choices = ['blue', 'grey', 'goldenrod', 'r', 'black', 'darkorange', 'g']

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    ax = ax.ravel() # Ravel turns a matrix into a vector... easier to iterate

    plt.subplots_adjust(bottom=0.00, top=1.00)

for i, column in enumerate(list_of_columns):
    ax[i].scatter(dataframe[column],
      dataframe[target_column],
      color=color_choices[i % len(color_choices)],
      alpha = 0.1)

# ax[i].set_title(f'{column} vs. {target_column}', fontsize=18)
ax[i].set_ylabel(f'{target_column}', fontsize=14)
ax[i].set_xlabel(f'{column}', fontsize=14)

if file:
    plt.savefig(file, bbox_inches='tight')
    plt.show();
return

# Unpickle Dataframe

with open('raw_data3/data/pre_processed_df.pkl', 'rb') as f:
    df = pickle.load(f)
    df.head()

    print(df.shape)
#(29067, 28)
    df.isna().sum().sum()

# Correlation Heatmap
    plt.figure(figsize=(15,15))
    sns.set(font_scale=1)

    mask = np.zeros_like(df.corr())
    mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):
    sns.heatmap(df.corr(), mask=mask, annot=False, cmap='coolwarm')
    plt.savefig('raw_data3/images/corr_heatmap.jpg', bbox_inches='tight');

    plt.figure(figsize=(4,10))
    sns.set(font_scale=1)
    sns.heatmap(df.corr()[['dam_price_per_mwh']].sort_values('dam_price_per_mwh', ascending=False),
                annot=True,
                cmap='coolwarm');
                plt.title('DAM Price Corrs', fontsize=14)
                plt.savefig('raw_data3/images/dam_corr_heatmap.jpg', bbox_inches='tight')
                plt.show();

    plt.figure(figsize=(4,10))
    sns.set(font_scale=1)
    sns.heatmap(df.corr()[['hasp_price_per_mwh']].sort_values('hasp_price_per_mwh', ascending=False),
                annot=True,
                cmap='coolwarm');
                plt.title('HASP Price Corrs', fontsize=14)
                
                plt.savefig('raw_data3/images/hasp_corr_heatmap.jpg', bbox_inches='tight')
                plt.show();

#***

# Income For a 1 MW Power Plant
# Adapted timedelta method from StackOverflow

# https://stackoverflow.com/questions/42521107/python-pandas-find-number-of-years-between-two-
dates

time_horiz_yrs = (df.index[-1] - df.index[0]) / timedelta(days=365)

print(time_horiz_yrs)
# 3.331164383561644

dam_ann_inc_1mw = df.dam_price_per_mwh.sum() / time_horiz_yrs
hasp_ann_inc_1mw = df.hasp_price_per_mwh.sum() / time_horiz_yrs

print(dam_ann_inc_1mw)
print(hasp_ann_inc_1mw)

# Data Visualization ... time series
import math
items_to_plot = df.columns[:24]
gen_linecharts(df, items_to_plot, 4, 'raw_data3/images/all_features_plots.jpg')

# Time series plots with one-week rolling averages
df = df.replace(to_replace=9999, method='ffill')
gen_linecharts_rolling(df, 24*7, items_to_plot, 4, 'raw_data3/images/all_feat_mv_avg.jpg')

#with open('raw_data3/data/df_outliers_rem.pkl', 'wb') as f:
# pickle.dump(df, f)

%matplotlib inline
sns.set_style('whitegrid')

sns.distplot(df.dam_price_per_mwh)

sns.distplot(df.hasp_price_per_mwh)

normaltest(df.dam_price_per_mwh)
# NormaltestResult(statistic=46658.68256692149, pvalue=0.0)

normaltest(df.hasp_price_per_mwh)

dam_features = [col for col in df.columns if col != 'dam_price_per_mwh']
print(dam_features)

hasp_features = [col for col in df.columns if col != 'hasp_price_per_mwh']

print(hasp_features)

gen_scatterplots(df, 'dam_price_per_mwh', dam_features, 4,
'raw_data3/images/dam_price_vs_all_scatters')

gen_scatterplots(df, 'hasp_price_per_mwh', hasp_features,
4,'raw_data3/images/hasp_price_vs_all_scatters')

# *** End of EDA and Inspection

# *** Start ARIMA_Model_Continuous_Targets

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score

def interpret_dftest(dftest):
dfoutput = pd.Series(dftest[0:2], index=['Test Statistic','p-value'])

return dfoutput

def MSE(true, predicted):
    squared_diff = np.square(true - predicted)
return np.mean(squared_diff)

# Root Mean Square Error
def RMSE(true, predicted):
    squared_diff = np.square(true - predicted)
return np.sqrt(np.mean(squared_diff))

# R-squared, coefficient of determination
def R_squared(true, predicted):
    true = np.array(true)
    predicted = np.array(predicted)
    sum_squared_diff = sum(np.square(true - predicted))
    variance = sum(np.square(true - np.mean(true)))
    calc_r2 = 1 - (sum_squared_diff / variance)
return calc_r2

# Load Pickles: Train/Test Dataframes & Full Dataframe

with open('raw_data3/data/processed/train.pkl', 'rb') as f:
    train = pickle.load(f)

with open('raw_data3/data/processed/test.pkl', 'rb') as f:
    test = pickle.load(f)

with open('raw_data3/data/pre_processed_df.pkl', 'rb') as f:
    df = pickle.load(f)

# Augmented Dickey-Fuller Test
# P-value for both target variables are extremely low
# (1.4e-19 and 6.5e-28 for day-ahead and hour-ahead prices, respectively),
# so they both pass the the test for stationarity, assuming some low
# threshold for $\alpha$ such as 0.05 or 0.01
# Differencing is not required, so set $d = 0$.

interpret_dftest(adfuller(train['dam_price_per_mwh']))

'''Test Statistic -1.086989e+01
p-value 1.377061e-19'''

interpret_dftest(adfuller(train['hasp_price_per_mwh']))

'''Test Statistic -1.516645e+01
p-value 6.481191e-28
dtype: float64'''

'''Choose Values for p and q
Endogenous variable: DA price ... $p=1$ ... sharp cutoff in PACF bet. lag-1 & lag-2; ACF lag-1 > 0
Endogenous variable: DA price ... $q=0$ ... ACF lag-1 is not negative

Endogenous variable: HA price ... $p=1$ ... sharp cutoff in PACF bet. lag-1 & lag-2; ACF lag-1 > 0
Endogenous variable: HA price ... $q=0$ ... ACF lag-1 is not negative'''

fig, ax = plt.subplots(figsize=(12,7))
plot_acf(train['dam_price_per_mwh'], lags=24*2, ax=ax)
plt.title('DAM Price Time Series ACF', fontsize=24)
plt.xlabel('Lag Intervals (Hours)', fontsize=18)
plt.savefig('raw_data3/images/dam_acf.jpg')
plt.show();

fig, ax = plt.subplots(figsize=(12,7))
plot_pacf(train['dam_price_per_mwh'], lags=24*2, ax=ax)
plt.title('DAM Price Time Series PACF', fontsize=24)
plt.xlabel('Lag Intervals (Hours)', fontsize=18)
plt.savefig('raw_data3/images/dam_pacf.jpg')
plt.show();

fig, ax = plt.subplots(figsize=(12,7))
plot_acf(train['hasp_price_per_mwh'], lags=24*2, ax=ax)
plt.title('HASP Price Time Series ACF', fontsize=24)
plt.xlabel('Lag Intervals (Hours)', fontsize=18)
plt.savefig('raw_data3/images/hasp_acf.jpg')
plt.show();

fig, ax = plt.subplots(figsize=(12,7))
plot_pacf(train['hasp_price_per_mwh'], lags=24*2, ax=ax)
plt.title('HASP Price Time Series PACF', fontsize=24)
plt.xlabel('Lag Intervals (Hours)', fontsize=18)
plt.savefig('raw_data3/images/hasp_pacf.jpg')
plt.show();

# *** skip dur cls this takes v long time***

# this will take a few hours
# v need to reduce the range(7) to lower val

# Gridsearch to find lowest MSE params for p, d, q
# (for DAM and HASP prices as endogenous variables)

num = 2
for p in range(num): #range(7):
    for d in range(num): #range(7):
        for q in range(num): #range(7):
            try:
                dam_arima = ARIMA(endog = train['dam_price_per_mwh'].astype('float32'), # Y variable
                                  order = (p, d, q)) # (p, d, q)
                dam_model = dam_arima.fit()
                dam_preds = dam_model.predict()
                print(f'MSE for (p={p}, d={d}, q={q}) ... {mean_squared_error(train["dam_price_per_mwh"],
                      dam_preds)}')

    except:
pass

'''
MSE for (p=2, d=0, q=0) ... 111.58563032669524

MSE for (p=2, d=0, q=1) ... 111.39900141012114

MSE for (p=2, d=0, q=2) ... 111.05483971028366

MSE for (p=2, d=0, q=3) ... 110.40302671708838

MSE for (p=2, d=0, q=4) ... 108.20296239918503

MSE for (p=2, d=0, q=5) ... 107.52114104117425

MSE for (p=2, d=0, q=6) ... 106.33272886535855

MSE for (p=3, d=0, q=0) ... 111.4848809500559

MSE for (p=3, d=0, q=1) ... 109.0750799883165

MSE for (p=3, d=0, q=2) ... 110.37585763606131

MSE for (p=3, d=0, q=3) ... 103.59016575273273

MSE for (p=3, d=0, q=4) ... 104.42594503751678

MSE for (p=3, d=0, q=5) ... 106.02240578341396

MSE for (p=4, d=0, q=0) ... 110.94492193346848

MSE for (p=4, d=0, q=1) ... 110.34219959803963

MSE for (p=4, d=0, q=2) ... 110.3051003650367

MSE for (p=4, d=0, q=3) ... 103.16989396155648

MSE for (p=4, d=0, q=4) ... 103.0663068398282

MSE for (p=5, d=0, q=0) ... 110.64704602369453

MSE for (p=5, d=0, q=1) ... 107.74096447094092

MSE for (p=5, d=0, q=2) ... 107.51383008826402

MSE for (p=5, d=0, q=3) ... 103.09368734837737

MSE for (p=5, d=0, q=4) ... 103.06572731127694

MSE for (p=5, d=0, q=5) ... 102.98192735976968

MSE for (p=5, d=0, q=6) ... 102.62335787227049

MSE for (p=6, d=0, q=0) ... 110.41546455091661

MSE for (p=6, d=0, q=1) ... 107.73739097027799

MSE for (p=6, d=0, q=2) ... 107.65320949360365

MSE for (p=6, d=0, q=3) ... 103.45801153274255

MSE for (p=6, d=0, q=4) ... 103.76103385643049

MSE for (p=6, d=0, q=5) ... 101.96864102481713

MSE for (p=6, d=0, q=6) ... 101.96709637187159
'''

# *** Do not run this dur cls ***
for p in range(num): #range(7):
for d in range(num): #range(7):
for q in range(num): #range(7):
try:
hasp_arima = ARIMA(endog = train['hasp_price_per_mwh'].astype('float32'), # Y variable
                   order = (p, d, q)) # (p, d, q)
    hasp_model = hasp_arima.fit()
    hasp_preds = hasp_model.predict()
    print(f'MSE for (p={p}, d={d}, q={q}) ... {mean_squared_error(train["hasp_price_per_mwh"],
          hasp_preds):.2f}')

    except:
        pass

MSE for (p=0, d=0, q=0) ... 1677.76
MSE for (p=0, d=0, q=1) ... 1159.63
MSE for (p=0, d=0, q=2) ... 1059.02
MSE for (p=0, d=0, q=3) ... 1032.19
MSE for (p=0, d=0, q=4) ... 1025.37
MSE for (p=0, d=0, q=5) ... 1021.57
MSE for (p=0, d=0, q=6) ... 1020.72
MSE for (p=1, d=0, q=0) ... 1021.64
MSE for (p=1, d=0, q=1) ... 1020.82
MSE for (p=1, d=0, q=2) ... 1020.76
MSE for (p=1, d=0, q=3) ... 1020.76
MSE for (p=1, d=0, q=4) ... 1020.75
MSE for (p=1, d=0, q=5) ... 1014.32
MSE for (p=1, d=0, q=6) ... 1012.21
MSE for (p=2, d=0, q=0) ... 1020.80
MSE for (p=2, d=0, q=1) ... 1020.76
MSE for (p=2, d=0, q=2) ... 1020.76
MSE for (p=2, d=0, q=3) ... 1012.68
MSE for (p=2, d=0, q=4) ... 1012.54
MSE for (p=2, d=0, q=5) ... 1012.50
MSE for (p=2, d=0, q=6) ... 1012.05
MSE for (p=3, d=0, q=0) ... 1020.76
MSE for (p=3, d=0, q=1) ... 1020.76

MSE for (p=3, d=0, q=2) ... 1010.34
MSE for (p=3, d=0, q=3) ... 1010.33
MSE for (p=3, d=0, q=4) ... 1012.50
MSE for (p=3, d=0, q=5) ... 1012.48
MSE for (p=3, d=0, q=6) ... 1013.46
MSE for (p=4, d=0, q=0) ... 1020.76
MSE for (p=4, d=0, q=1) ... 1012.58
MSE for (p=4, d=0, q=2) ... 1010.33
MSE for (p=4, d=0, q=3) ... 1009.92
MSE for (p=4, d=0, q=4) ... 1001.51
MSE for (p=4, d=0, q=5) ... 1001.00
MSE for (p=4, d=0, q=6) ... 994.45
MSE for (p=5, d=0, q=0) ... 1020.76
MSE for (p=5, d=0, q=1) ... 1012.39
MSE for (p=5, d=0, q=2) ... 1012.41
MSE for (p=5, d=0, q=3) ... 1009.89
MSE for (p=5, d=0, q=4) ... 1000.91
MSE for (p=5, d=0, q=5) ... 1000.88
MSE for (p=5, d=0, q=6) ... 992.86
MSE for (p=6, d=0, q=0) ... 1020.61
MSE for (p=6, d=0, q=1) ... 1012.22
MSE for (p=6, d=0, q=2) ... 1012.36
MSE for (p=6, d=0, q=3) ... 1009.64
MSE for (p=6, d=0, q=4) ... 1000.88
MSE for (p=6, d=0, q=5) ... 999.73

MSE for (p=6, d=0, q=6) ... 991.40

# Instantiate and Fit ARIMA for Day Ahead with 4, 0, 6 as Params
p = 4
d = 0
q = 6
dam_arima406 = ARIMA(endog = train['dam_price_per_mwh'].astype('float32'), # Y variable
order = (p, d, q)) # (p, d, q)
dam_arima406_model = dam_arima406.fit()
dam_arima406_model.summary()

ARMA Model Results
Dep. Variable: dam_price_per_mwh No. Observations: 21796
Model: ARMA(4, 6) Log Likelihood -81253.837
Method: css-mle S.D. of innovations 10.063
Date: Sun, 14 Jul 2019 AIC 162531.674
Time: 15:25:46 BIC 162627.548
Sample:0 HQIC 162562.911
coef std err z P>|z| [0.025 0.975]
const 32.7567 1.406 23.300 0.000 30.001 35.512
ar.L1.dam_price_per_mwh 3.3247 0.005 652.745 0.000 3.315 3.335
ar.L2.dam_price_per_mwh -4.3551 0.014 -315.687 0.000 -4.382 -4.328
ar.L3.dam_price_per_mwh 2.6269 0.014 191.486 0.000 2.600 2.654
ar.L4.dam_price_per_mwh -0.5969 0.005 -119.353 0.000 -0.607 -0.587
ma.L1.dam_price_per_mwh -2.2640 0.010 -219.670 0.000 -2.284 -2.244
ma.L2.dam_price_per_mwh 1.5790 0.024 65.428 0.000 1.532 1.626
ma.L3.dam_price_per_mwh 0.0550 0.023 2.375 0.018 0.010 0.100
ma.L4.dam_price_per_mwh -0.4261 0.020 -20.792 0.000 -0.466 -0.386
ma.L5.dam_price_per_mwh 0.1413 0.017 8.192 0.000 0.107 0.175
ma.L6.dam_price_per_mwh -0.0771 0.007 -11.484 0.000 -0.090 -0.064
Roots
Real Imaginary Modulus Frequency
AR.1 0.8661 -0.5025j 1.0013 -0.0837
AR.2 0.8661 +0.5025j 1.0013 0.0837
AR.3 1.0036 -0.0000j 1.0036 -0.0000
AR.4 1.6648 -0.0000j 1.6648 -0.0000

MA.1 1.0194 -0.0000j 1.0194 -0.0000
MA.2 0.8758 -0.5068j 1.0118 -0.0835
MA.3 0.8758 +0.5068j 1.0118 0.0835
MA.4 -1.8057 -0.0000j 1.8057 -0.5000
MA.5 0.4340 -2.5881j 2.6242 -0.2236
MA.6 0.4340 +2.5881j 2.6242 0.2236

print(f'df: {df.shape[0]}')
print(f'train: {train.shape[0]}')
print(f'test: {test.shape[0]}')
print(f'trn+tst:{train.shape[0] + test.shape[0]}')
'''df: 29067
train: 21800
test: 7267
trn+tst:29067'''

dam_arima406_train_preds = dam_arima406_model.predict(start= 0,
end = train.shape[0]-1)
print(f'MSE Train ARIMA({p},{d},{q}) ... \
{mean_squared_error(train["dam_price_per_mwh"], dam_arima406_train_preds):.2f}')

print(f'R-sq Train ARIMA({p},{d},{q}) ... \
{r2_score(train["dam_price_per_mwh"], dam_arima406_train_preds):.4f}')

# MSE Train ARIMA(4,0,6) ... 101.24
# R-sq Train ARIMA(4,0,6) ... 0.7625

dam_arima406_test_preds = dam_arima406_model.predict(start = train.shape[0],
                                                     end = df.shape[0]-1)
    print(f'MSE Test ARIMA({p},{d},{q}) ... \
          {mean_squared_error(test["dam_price_per_mwh"], dam_arima406_test_preds):.2f}')

    print(f'R-sq Test ARIMA({p},{d},{q}) ... \
{r2_score(test["dam_price_per_mwh"], dam_arima406_test_preds):.4f}')

# MSE Test ARIMA(4,0,6) ... 1692.87
# R-sq Test ARIMA(4,0,6) ... -0.1732

dam_arima406_train_resid = train["dam_price_per_mwh"] - dam_arima406_train_preds
dam_arima406_test_resid = test["dam_price_per_mwh"] - dam_arima406_test_preds

plt.plot(dam_arima406_train_resid);
'''df: 29062
train: 21796
test: 7266
trn+tst:29062
MSE Train ARIMA(4,0,6) ... 101.27
R-sq Train ARIMA(4,0,6) ... 0.7625

MSE Test ARIMA(4,0,6) ... 1690.13
R-sq Test ARIMA(4,0,6) ... -0.1712'''

with open('raw_data3/fitted_models/dam_arima406_model.pkl', 'wb') as f:
    pickle.dump(dam_arima406_model, f)

with open('raw_data3/data/predictions/dam_arima406_train_preds.pkl', 'wb') as f:
    pickle.dump(dam_arima406_train_preds, f)

with open('raw_data3/data/predictions/dam_arima406_test_preds.pkl', 'wb') as f:
    pickle.dump(dam_arima406_test_preds, f)

# Start running from here
# Instantiate and Fit ARIMA for Hour Ahead with 6, 0, 6 as Params
'''x = 6
y = 0
z = 6'''
x = 1
y = 0
z = 1
hasp_arima606 = ARIMA(endog = train['hasp_price_per_mwh'].astype('float32'), # Y variable
order = (x, y, z)) # (p, d, q)
hasp_arima606_model = hasp_arima606.fit()
hasp_arima606_model.summary()

'''
ARMA Model Results

Dep. Variable: hasp_price_per_mwh No. Observations: 21796
Model: ARMA(6, 6) Log Likelihood -106114.856
Method: css-mle S.D. of innovations 31.486
Date: Tue, 16 Jul 2019 AIC 212257.713
Time: 15:18:25 BIC 212369.566
Sample:0 HQIC 212294.156
coef std err z P>|z| [0.025 0.975]
const 32.0610 2.052 15.624 0.000 28.039 36.083
ar.L1.hasp_price_per_mwh 2.4369 0.016 156.086 0.000 2.406 2.468
ar.L2.hasp_price_per_mwh -2.4345 0.035 -69.184 0.000 -2.503 -2.366
ar.L3.hasp_price_per_mwh 2.1030 0.033 63.720 0.000 2.038 2.168
ar.L4.hasp_price_per_mwh -2.5205 0.046 -55.198 0.000 -2.610 -2.431
ar.L5.hasp_price_per_mwh 1.9427 0.041 46.920 0.000 1.862 2.024
ar.L6.hasp_price_per_mwh -0.5283 0.015 -36.183 0.000 -0.557 -0.500
ma.L1.hasp_price_per_mwh -1.8172 0.017 -108.248 0.000 -1.850 -1.784
ma.L2.hasp_price_per_mwh 1.2875 0.032 40.591 0.000 1.225 1.350
ma.L3.hasp_price_per_mwh -1.2747 0.026 -48.747 0.000 -1.326 -1.223
ma.L4.hasp_price_per_mwh 1.7067 0.034 49.829 0.000 1.640 1.774
ma.L5.hasp_price_per_mwh -0.8351 0.028 -29.670 0.000 -0.890 -0.780
ma.L6.hasp_price_per_mwh -0.0605 0.011 -5.337 0.000 -0.083 -0.038
Roots
Real Imaginary Modulus Frequency
AR.1 -0.4361 -0.9209j 1.0190 -0.3204
AR.2 -0.4361 +0.9209j 1.0190 0.3204
AR.3 0.8659 -0.5023j 1.0010 -0.0837

AR.4 0.8659 +0.5023j 1.0010 0.0837
AR.5 1.0020 -0.0000j 1.0020 -0.0000
AR.6 1.8156 -0.0000j 1.8156 -0.0000
MA.1 -0.4297 -0.9211j 1.0164 -0.3195
MA.2 -0.4297 +0.9211j 1.0164 0.3195
MA.3 0.8701 -0.5041j 1.0055 -0.0836
MA.4 0.8701 +0.5041j 1.0055 0.0836
MA.5 1.0083 -0.0000j 1.0083 -0.0000
MA.6 -15.6891 -0.0000j 15.6891 -0.5000 '''

hasp_arima606_train_preds = hasp_arima606_model.predict(start= 0,
                                                        end = train.shape[0]-1)
    print(f'MSE Train ARIMA({x},{y},{z}) ... \
          {MSE(train["hasp_price_per_mwh"], hasp_arima606_train_preds):.2f}')

    print(f'R-sq Train ARIMA({x},{y},{z}) ... \
          {R_squared(train["hasp_price_per_mwh"], hasp_arima606_train_preds):.4f}')

# MSE Train ARIMA(6,0,6) ... 991.07
# R-sq Train ARIMA(6,0,6) ... 0.4092

    hasp_arima606_test_preds = hasp_arima606_model.predict(start = train.shape[0],
                                                           end = df.shape[0]-1)
    print(f'MSE Test ARIMA({x},{y},{z}) ... \
          {MSE(test["hasp_price_per_mwh"], hasp_arima606_test_preds):.2f}')

    print(f'R-sq Test ARIMA({x},{y},{z}) ... \
          {R_squared(test["hasp_price_per_mwh"], hasp_arima606_test_preds):.4f}')

# MSE Test ARIMA(6,0,6) ... nan
# R-sq Test ARIMA(6,0,6) ... -0.1333

with open('raw_data3/fitted_models/hasp_arima606_model.pkl', 'wb') as f:
    pickle.dump(hasp_arima606_model, f)

with open('raw_data3/data/predictions/hasp_arima606_train_preds.pkl', 'wb') as f:
    pickle.dump(hasp_arima606_train_preds, f)

with open('raw_data3/data/predictions/hasp_arima606_test_preds.pkl', 'wb') as f:
    pickle.dump(hasp_arima606_test_preds, f)

'''

MSE Train ARIMA(6,0,6) ... 991.40
R-sq Train ARIMA(6,0,6) ... 0.4091

MSE Test ARIMA(6,0,6) ... nan
R-sq Test ARIMA(6,0,6) ... -0.1427
'''

# *** End of 6a_ARIMA_Model_Continuous_Targets

# *** Start SARIMAX_Model_Cont_Targets_scaled

# Load Pickles: Train/Test Dataframes & Scaled Arrays
with open('raw_data3/data/processed/train_sc_df.pkl', 'rb') as f:
    train_sc_df = pickle.load(f)

with open('raw_data3/data/processed/test_sc_df.pkl', 'rb') as f:
    test_sc_df = pickle.load(f)

with open('raw_data3/data/pre_processed_df.pkl', 'rb') as f:
    df = pickle.load(f)

with open('raw_data3/data/processed/train.pkl', 'rb') as f:
    train = pickle.load(f)

with open('raw_data3/data/processed/test.pkl', 'rb') as f:
    test = pickle.load(f)

exog_col_for_dam = [train.columns[i] for i, col in enumerate(df.columns) if col != 'dam_price_per_mwh']
    exog_col_for_hasp = [train.columns[i] for i, col in enumerate(df.columns) if col !=
                         'hasp_price_per_mwh']

X_train_sc_dam = train_sc_df[exog_col_for_dam]
X_test_sc_dam = test_sc_df[exog_col_for_dam]
y_train_dam = train['dam_price_per_mwh']
y_test_dam = test['dam_price_per_mwh']

X_train_sc_hasp = train_sc_df[exog_col_for_hasp]
X_test_sc_hasp = test_sc_df[exog_col_for_hasp]
y_train_hasp = train['hasp_price_per_mwh']
y_test_hasp = test['hasp_price_per_mwh']

print(y_train_dam.shape)
#(21800,)

print(X_train_sc_dam.shape)
#(21800, 27)

''' Gridsearch to find best MSE params for P, D, Q, and S
(for DAM and HASP prices as endogenous variables,
use the ARIMA parameters for p, d, & q found in previous ARIMA gridsearch...
DAM: 4, 0, 6 and HASP: 6, 0 ,6)

Because of the amount of time each SARIMAX model takes to fit,
there was simply not enough time to do these grid searches,
the code is correct and ready to run'''

# Starting MSE and (P, D, Q).

# mse = 99 * (10 ** 16)
# final_P = 0
# final_D = 0
# final_Q = 0
# final_S = 0

# for P in range(3):
# for Q in range(3):
# for D in range(3):
# for S in range(0,24,8):
# try:
# # Instantiate SARIMA model.
# sarima = SARIMAX(endog = y_train_dam,
# order = (4, 0, 6), # (p, d, q)
# seasonal_order = (P, D, Q, S), # (P, D, Q, S)
# exog = X_train_sc_dam)

# # Fit SARIMA model.
# model = sarima.fit()

# # Generate predictions based on training set.
# preds = model.predict()

# # Evaluate predictions.
# print(f'MSE for (4,0,6) x ({P},{D},{Q},{S}) ...
{mean_squared_error(train["dam_price_per_mwh"], preds)}')

# # Save for final report.
# if mse > mean_squared_error(train['dam_price_per_mwh'], preds):
# mse = mean_squared_error(train['dam_price_per_mwh'], preds)
# final_P = P

# final_D = D
# final_Q = Q
# final_S = S

# except:
# pass

# print(f'Our model that minimizes MSE on the training data is the SARIMA(4,0,6) x
({final_P},{final_D},{final_Q},{final_S})')
# print(f'This model MSE = {mse}')

# Starting MSE and (P, D, Q).

# mse = 99 * (10 ** 16)
# final_P = 0
# final_D = 0
# final_Q = 0
# final_S = 0

# for P in range(2):
# for Q in range(2):
# for D in range(2):
# for S in range(0,25,12):
# try:
# # Instantiate SARIMA model.
# sarima = SARIMAX(endog = y_train_hasp],
# order = (6, 0, 6), # (p, d, q)
# seasonal_order = (P, D, Q, S), # (P, D, Q, S)
# exog = X_train_sc_hasp)

# # Fit SARIMA model.
# model = sarima.fit()

# # Generate predictions based on training set.
# preds = model.predict()

# # Evaluate predictions.
# print(f'MSE for (6,0,6) x ({P},{D},{Q},{S}) ...
{mean_squared_error(train["hasp_price_per_mwh"], preds)}')

# # Save for final report.
# if mse > mean_squared_error(train['hasp_price_per_mwh'], preds):
# mse = mean_squared_error(train['hasp_price_per_mwh'], preds)
# final_P = P

# final_D = D
# final_Q = Q
# final_S = S

# except:
# pass

# print(f'Our model that minimizes MSE on the training data is the SARIMA(6,0,6) x
({final_P},{final_D},{final_Q},{final_S})')
# print(f'This model MSE = {mse}')

# Instantiate and fit the models with best params
P = 0
D = 1
Q = 0
S = 24
# Instantiate SARIMA model.
dam_sarimax01024 = sm.tsa.statespace.SARIMAX(endog = y_train_dam,
                                             order = (4, 0, 6), # (p, d, q)
                                             seasonal_order = (P, D, Q, S), # (P, D, Q, S)
                                             exog = X_train_sc_dam)
# Fit SARIMA model.
    dam_sarimax01024_model = dam_sarimax01024.fit()
# Generate predictions based on training set.
    dam_sarimax01024_preds = dam_sarimax01024_model.predict()
# Evaluate predictions.
    print(f'MSE for (4,0,6) x ({P},{D},{Q},{S}) ... {mean_squared_error(train["dam_price_per_mwh"],
          dam_sarimax01024_preds):.2f}')

with open('raw_data3/fitted_models/dam_sarimax01024_model.pkl', 'wb') as f:
    pickle.dump(dam_sarimax01024_model, f)

with open('raw_data3/data/predictions/dam_sarimax01024_preds.pkl', 'wb') as f:
    pickle.dump(dam_sarimax01024_preds, f)
    
    P = 0
    D = 1
    Q = 0
    S = 24
# Instantiate SARIMA model.
    hasp_sarimax01024 = SARIMAX(endog = y_train_hasp,
                                order = (6, 0, 6), # (p, d, q)
                                seasonal_order = (P, D, Q, S), # (P, D, Q, S)
                                exog = X_train_sc_hasp)

# Fit SARIMA model.
    hasp_sarimax01024_model = hasp_sarimax01024.fit()
    
# Generate predictions based on training set.
    hasp_sarimax01024_preds = hasp_sarimax01024_model.predict()

# Evaluate predictions.
    print(f'MSE for (6,0,6) x ({P},{D},{Q},{S}) ... {mean_squared_error(train["hasp_price_per_mwh"],
          hasp_sarimax01024_preds):.2f}')
# MSE for (6,0,6) x (0,1,0,24) ... 1497.90

with open('raw_data3/fitted_models/hasp_sarimax01024_model.pkl', 'wb') as f:
    pickle.dump(hasp_sarimax01024_model, f)

with open('raw_data3/data/predictions/hasp_sarimax01024_preds.pkl', 'wb') as f:
    pickle.dump(hasp_sarimax01024_preds, f)

# *** End of 6b_SARIMAX_Model_Cont_Targets_scaled

# *** Start RNN_Model_DAM_20epochs_Lookbk12.ipynb

# RNN

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.optimizers import Adam
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import regularizers

#Functions
def MSE(true, predicted):
    squared_diff = np.square(true - predicted)
return np.mean(squared_diff)

# Root Mean Square Error
def RMSE(true, predicted):
    squared_diff = np.square(true - predicted)
return np.sqrt(np.mean(squared_diff))

# R-squared, coefficient of determination
def R_squared(true, predicted):
    true = np.array(true)
    predicted = np.array(predicted)
    sum_squared_diff = sum(np.square(true - predicted))
    variance = sum(np.square(true - np.mean(true)))

    calc_r2 = 1 - (sum_squared_diff / variance)
return calc_r2

# Load Pickles: Train/Test Dataframes & Scaled Arrays
with open('raw_data3/data/processed/train.pkl', 'rb') as f:
    train = pickle.load(f)

with open('raw_data3/data/processed/test.pkl', 'rb') as f:
    test = pickle.load(f)

# Create (scaled) Train-test X and y Pairs for Day Ahead RNN Model
    features = [col for col in train.columns if col != 'dam_price_per_mwh']
    print(len(features))
    print(features)
# 27
# ['hasp_price_per_mwh', 'rtm_price_per_mwh', '7da_load_fcast_mw', '2da_load_fcast_mw',
'dam_load_fcast_mw', 'rtm_load_fcast_mw', 'water_acre_feet', 'sand_temp', 'sand_wind', 'sand_vis',
'sand_ceil', 'fres_temp', 'fres_wind', 'fres_vis', 'fres_ceil', 'rive_temp', 'rive_wind', 'rive_vis', 'rive_ceil',
'redd_temp', 'redd_wind', 'redd_vis', 'redd_ceil', 'year', 'month', 'day', 'hour']

X_train = train[features]
y_train = train['dam_price_per_mwh']

X_test = test[features]
y_test = test['dam_price_per_mwh']

X_train.shape
# (21800, 27)

ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

# Time Series Generator
# RNN Time Series Generator Parameters
lookback_length = 12 # one week look-back
batch_size = 64

train_sequences = TimeseriesGenerator(X_train_sc,
                                      y_train,
                                      length=lookback_length,
                                      batch_size=batch_size)
        train_sequences

    batch_x, batch_y = train_sequences[0]
    
    test_sequences = TimeseriesGenerator(X_test_sc,y_test,length=lookback_length,batch_size=batch_size)

len(train_sequences)
# 341

train_sequences[0][0].shape
# (64, 12, 27)

print(batch_x.shape)
print(batch_y.shape)
#(64, 12, 27)
#(64,)

# RNN Model
model = Sequential()

model.add(GRU(16,
              input_shape=(batch_x.shape[1],
             batch_x.shape[2]),
             return_sequences=True))
    model.add(GRU(16))
    model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1)) # refers to nodes in the first hidden layer
    model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.1))
    model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(lr=.0005),
              loss='mean_squared_error')

    history = model.fit_generator(train_sequences,
                                  validation_data=test_sequences,
                                  epochs=10, #20,
                                  verbose=1)

'''WARNING:tensorflow:From /Users/owner/anaconda3/envs/ds2/lib/python3.6/site-
packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops)

is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.

Epoch 1/20
341/341 [==============================] - 8s 24ms/step - loss: 1179.8026 - val_loss: 1740.3277
Epoch 2/20
341/341 [==============================] - 7s 19ms/step - loss: 418.5544 - val_loss: 1691.7571
Epoch 3/20
341/341 [==============================] - 7s 20ms/step - loss: 386.0040 - val_loss: 1670.8257
Epoch 4/20
341/341 [==============================] - 7s 20ms/step - loss: 337.2425 - val_loss: 1393.9671
Epoch 5/20
341/341 [==============================] - 7s 20ms/step - loss: 296.8123 - val_loss: 1268.2520
Epoch 6/20
341/341 [==============================] - 9s 27ms/step - loss: 272.2944 - val_loss: 1201.5058
Epoch 7/20
341/341 [==============================] - 7s 21ms/step - loss: 256.6564 - val_loss: 1126.7057
Epoch 8/20
341/341 [==============================] - 7s 20ms/step - loss: 239.9857 - val_loss: 1086.4520
Epoch 9/20
341/341 [==============================] - 6s 19ms/step - loss: 224.1302 - val_loss: 1039.8475
Epoch 10/20
341/341 [==============================] - 7s 21ms/step - loss: 212.6751 - val_loss: 1022.0788
Epoch 11/20
341/341 [==============================] - 7s 20ms/step - loss: 197.3020 - val_loss: 911.8190
Epoch 12/20
341/341 [==============================] - 7s 19ms/step - loss: 190.5990 - val_loss: 915.6796
Epoch 13/20

341/341 [==============================] - 7s 20ms/step - loss: 179.5243 - val_loss: 868.1363
Epoch 14/20
341/341 [==============================] - 7s 20ms/step - loss: 172.3146 - val_loss: 838.6682
Epoch 15/20
341/341 [==============================] - 7s 20ms/step - loss: 163.4735 - val_loss: 856.5791
Epoch 16/20
341/341 [==============================] - 7s 20ms/step - loss: 152.3535 - val_loss: 835.2529
Epoch 17/20
341/341 [==============================] - 7s 20ms/step - loss: 148.9531 - val_loss: 785.5258
Epoch 18/20
341/341 [==============================] - 7s 20ms/step - loss: 140.9169 - val_loss: 873.2137
Epoch 19/20
341/341 [==============================] - 7s 20ms/step - loss: 141.8063 - val_loss: 841.1208
Epoch 20/20
341/341 [==============================] - 7s 21ms/step - loss: 127.6876 - val_loss: 792.6214
'''

train_loss = history.history['loss']
test_loss = history.history['val_loss']

plt.figure(figsize=(15, 8))
plt.plot(train_loss, label='Training MSE', color='darkblue')
plt.plot(test_loss, label='Testing MSE', color='darkorange')

plt.title('DAM RNN History ... 20 epochs, lookbk=12 hrs', fontsize=24)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss: Mean Squared Error', fontsize=18)
plt.legend()
plt.savefig('raw_data3/images/dam_RNN_12_lkbk.jpg', bbox_inches='tight')
plt.show();

with open('raw_data3/fitted_models/dam_rnn_model_12_lkbk.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('raw_data3/data/predictions/dam_rnn_history_12_lkbk.pkl', 'wb') as f:
    pickle.dump(history, f)
# Evaluate
# Unused but useful code
# x_axis = X_train.index[lookback_length:]
# y_hat = model.predict_generator(train_sequences)

pred_train = []
y_train_true = []
for i in range(len(train_sequences)):
    pred_train.extend(model.predict(train_sequences[i][0]).ravel())
    y_train_true.extend(train_sequences[i][1].ravel())
    print(f'Train pred: {len(pred_train)} rows')
    print(f'Train y-true: {len(y_train_true)} rows')
# Train pred: 21788 rows
# Train y-true: 21788 rows

pred_test = []
y_test_true = []
for i in range(len(test_sequences)):
    pred_test.extend(model.predict(test_sequences[i][0]).ravel())
    y_test_true.extend(test_sequences[i][1].ravel())
    print(f'Test pred: {len(pred_test)} rows')
    print(f'Test y-true: {len(y_test_true)} rows')
# Test pred: 7255 rows
# Test y-true: 7255 rows
    print(f'Train R2 score: {R_squared(y_train_true, pred_train):.4f}\
                         \nTest R2 score {R_squared(y_test_true, pred_test):.4f}\
                         \nOverfit: {R_squared(y_train_true, pred_train) - R_squared(y_test_true, pred_test):.4f}')
# Train R2 score: 0.7324
# Test R2 score 0.4511
# Overfit: 0.2813

dam_RNN_train_residuals = [y_train_true[i] - pred_train[i] for i in range(len(y_train_true))]
dam_RNN_test_residuals = [y_test_true[i] - pred_test[i] for i in range(len(y_test_true))]

plt.figure(figsize=(15,8))

plt.plot(y_train_true,
         label='train y-true',
         alpha = 0.8,
         color='darkblue')
    plt.plot(pred_train,
             label='train predictions',
             alpha = 0.5,
             color='darkorange')
plt.legend()
plt.title('DAM - RNN_12 Train: preds vs true (all)', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_12_train_pred_vs_true_all.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

# plt.plot(y_train[3:])
plt.plot(y_train_true,
         label='train y-true',
         alpha = 0.8,
         color='darkblue')
    plt.plot(pred_train,
             label='train predictions',
             alpha = 0.6,
             color='darkorange')
plt.legend()
plt.xlim(13_500, 14_750)
plt.title('DAM - RNN_12 Train: preds vs true (zoomed2)', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_12_train_pred_vs_true_zoomed2.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(y_test_true,
         label='test y-true',
         alpha = 0.8,
         color='grey')
    plt.plot(pred_test,
             label='test predictions',
             alpha = 0.5,
             color='goldenrod')
plt.legend()
plt.xlim(600, 1200)
plt.title('DAM - RNN_12 Test: preds vs true (zoomed)', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_12_test_pred_vs_true_zoomed.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(dam_RNN_train_residuals,
         label='train_residuals',
         alpha = 0.9,
         color='darkred')
# plt.legend()
plt.title('DAM RNN_12 Residual Errors - Train', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_12_train_residuals.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(dam_RNN_test_residuals,
         label='test_residuals',
         alpha = 0.9,
         color='darkorange')
# plt.legend()
plt.title('DAM RNN_12 Residual Errors - Test', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_12_test_residuals.jpg', bbox_inches='tight')
plt.show();

# *** End of 7a_RNN_Model_DAM_20epochs_Lookbk12

'''Functions
def MSE(true, predicted):
squared_diff = np.square(true - predicted)
return np.mean(squared_diff)

# Root Mean Square Error
def RMSE(true, predicted):
squared_diff = np.square(true - predicted)
return np.sqrt(np.mean(squared_diff))

# R-squared, coefficient of determination
def R_squared(true, predicted):
true = np.array(true)
predicted = np.array(predicted)
sum_squared_diff = sum(np.square(true - predicted))
variance = sum(np.square(true - np.mean(true)))
calc_r2 = 1 - (sum_squared_diff / variance)
return calc_r2
'''
# *** Start RNN_Model_DAM_20epochs_Lookbk168
# Load Pickles: Train/Test Dataframes & Scaled Arrays
with open('raw_data3/data/processed/train.pkl', 'rb') as f:
    train = pickle.load(f)

with open('raw_data3/data/processed/test.pkl', 'rb') as f:
    test = pickle.load(f)

# Create (scaled) Train-test X and y Pairs for Day Ahead RNN Model
features = [col for col in train.columns if col != 'dam_price_per_mwh']
    print(len(features))
    print(features)
# 27
# ['hasp_price_per_mwh', 'rtm_price_per_mwh', '7da_load_fcast_mw', '2da_load_fcast_mw',
'dam_load_fcast_mw', 'rtm_load_fcast_mw', 'water_acre_feet', 'sand_temp', 'sand_wind', 'sand_vis',

'sand_ceil', 'fres_temp', 'fres_wind', 'fres_vis', 'fres_ceil', 'rive_temp', 'rive_wind', 'rive_vis', 'rive_ceil',
'redd_temp', 'redd_wind', 'redd_vis', 'redd_ceil', 'year', 'month', 'day', 'hour']

X_train = train[features]
y_train = train['dam_price_per_mwh']

X_test = test[features]
y_test = test['dam_price_per_mwh']

X_train.shape
# (21800, 27)

ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

# Time Series Generator
# RNN Time Series Generator Parameters
lookback_length = 24*7 # one week look-back = 168 hours
batch_size = 64

train_sequences = TimeseriesGenerator(X_train_sc,
                                      y_train,
                                      length=lookback_length,
                                      batch_size=batch_size)
    batch_x, batch_y = train_sequences[0]

test_sequences = TimeseriesGenerator(X_test_sc,
                                     y_test,
                                     length=lookback_length,
                                     batch_size=batch_size)

    len(train_sequences)
# 338

    train_sequences[0][0].shape
# (64, 168, 27)

    print(batch_x.shape)
    print(batch_y.shape)
# (64, 168, 27)
# (64,)

# RNN Model
model = Sequential()model.add(GRU(16,
                  input_shape=(batch_x.shape[1],
                               batch_x.shape[2]),
return_sequences=True))
model.add(GRU(16))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1)) # refers to nodes in the first hidden layer
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(lr=.0005),loss='mean_squared_error')
history = model.fit_generator(train_sequences,
                              validation_data=test_sequences,
                              epochs=20,
                              verbose=1)
'''Use tf.cast instead.
Epoch 1/20
338/338 [==============================] - 86s 255ms/step - loss: 1109.0859 - val_loss: 1803.6559
Epoch 2/20
338/338 [==============================] - 85s 251ms/step - loss: 403.1389 - val_loss: 1618.8182
Epoch 3/20
338/338 [==============================] - 84s 249ms/step - loss: 367.9987 - val_loss: 1557.0463
Epoch 4/20
338/338 [==============================] - 88s 261ms/step - loss: 346.2294 - val_loss: 1441.5281
Epoch 5/20
338/338 [==============================] - 89s 263ms/step - loss: 331.6280 - val_loss: 1389.9252
Epoch 6/20
338/338 [==============================] - 87s 259ms/step - loss: 317.1309 - val_loss: 1300.2319
Epoch 7/20
338/338 [==============================] - 86s 253ms/step - loss: 307.0112 - val_loss: 1237.2437
Epoch 8/20
338/338 [==============================] - 86s 255ms/step - loss: 292.4397 - val_loss: 1225.3131
Epoch 9/20
338/338 [==============================] - 85s 252ms/step - loss: 275.6683 - val_loss: 1150.0790

Epoch 10/20
338/338 [==============================] - 82s 243ms/step - loss: 262.7249 - val_loss: 1076.8976
Epoch 11/20
338/338 [==============================] - 83s 246ms/step - loss: 249.4088 - val_loss: 1026.5813
Epoch 12/20
338/338 [==============================] - 85s 252ms/step - loss: 237.9790 - val_loss: 957.2217
Epoch 13/20
338/338 [==============================] - 85s 252ms/step - loss: 229.1679 - val_loss: 957.0442
Epoch 14/20
338/338 [==============================] - 81s 238ms/step - loss: 220.7759 - val_loss: 944.0775
Epoch 15/20
338/338 [==============================] - 84s 248ms/step - loss: 212.3017 - val_loss: 907.7619
Epoch 16/20
338/338 [==============================] - 85s 252ms/step - loss: 201.8687 - val_loss: 890.5509
Epoch 17/20
338/338 [==============================] - 84s 248ms/step - loss: 189.5289 - val_loss: 923.3543
Epoch 18/20
338/338 [==============================] - 90s 265ms/step - loss: 192.8715 - val_loss: 855.3536
Epoch 19/20
338/338 [==============================] - 86s 254ms/step - loss: 183.5713 - val_loss: 860.5396
Epoch 20/20
338/338 [==============================] - 85s 253ms/step - loss: 174.5455 - val_loss: 816.2674
'''

train_loss = history.history['loss']
test_loss = history.history['val_loss']

plt.figure(figsize=(15, 8))
plt.plot(train_loss, label='Training MSE', color='darkblue')
plt.plot(test_loss, label='Testing MSE', color='darkorange')

plt.title('DAM RNN History ... 20 epochs, lookbk=168 hrs', fontsize=24)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss: Mean Squared Error', fontsize=18)
plt.legend()
plt.savefig('raw_data3/images/dam_RNN_168_lkbk.jpg', bbox_inches='tight')
plt.show();

with open('raw_data3/fitted_models/dam_rnn_model_168_lkbk.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('raw_data3/data/predictions/dam_rnn_history_168_lkbk.pkl', 'wb') as f:
    pickle.dump(history, f)

# Evaluate
pred_train = []
y_train_true = []

for i in range(len(train_sequences)):
    pred_train.extend(model.predict(train_sequences[i][0]).ravel())
    y_train_true.extend(train_sequences[i][1].ravel())

    print(f'Train pred: {len(pred_train)} rows')
    print(f'Train y-true: {len(y_train_true)} rows')
# Train pred: 21632 rows
# Train y-true: 21632 rows

pred_test = []
y_test_true = []

for i in range(len(test_sequences)):
    pred_test.extend(model.predict(test_sequences[i][0]).ravel())
    y_test_true.extend(test_sequences[i][1].ravel())

    print(f'Test pred: {len(pred_test)} rows')
    print(f'Test y-true: {len(y_test_true)} rows')
# Test pred: 7099 rows
# Test y-true: 7099 rows

print(f'Train R2 score: {R_squared(y_train_true, pred_train):.4f}\
                         \nTest R2 score {R_squared(y_test_true, pred_test):.4f}\
                         \nOverfit: {R_squared(y_train_true, pred_train) - R_squared(y_test_true, pred_test):.4f}')
# Train R2 score: 0.6037
# Test R2 score 0.4395
# Overfit: 0.1642

dam_RNN_train_residuals = [y_train_true[i] - pred_train[i] for i in range(len(y_train_true))]
dam_RNN_test_residuals = [y_test_true[i] - pred_test[i] for i in range(len(y_test_true))]

plt.figure(figsize=(15,8))

plt.plot(y_train_true,
         label='train y-true',
         alpha = 0.8,
         color='darkblue')

plt.plot(pred_train,
         label='train predictions',
         alpha = 0.5,
         color='darkorange')
plt.legend()
plt.title('DAM - RNN_168 Train: preds vs true (all)', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_168_train_pred_vs_true_all.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

# plt.plot(y_train[3:])
plt.plot(y_train_true,
         label='train y-true',
         alpha = 0.8,
         color='darkblue')
    plt.plot(pred_train,
             label='train predictions',
             alpha = 0.6,
             color='darkorange')
plt.legend()
plt.xlim(13_500, 14_750)
plt.title('DAM - RNN_168 Train: preds vs true (zoomed2)', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_168_train_pred_vs_true_zoomed2.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(y_test_true,
         label='test y-true',
         alpha = 0.8,
         color='grey')
    plt.plot(pred_test,
             label='test predictions',
             alpha = 0.5,
             color='goldenrod')
plt.legend()
plt.xlim(600, 1200)
plt.title('DAM - RNN_168 Test: preds vs true (zoomed)', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_168_test_pred_vs_true_zoomed.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(dam_RNN_train_residuals,
         label='train_residuals',
         alpha = 0.9,
         color='darkred')
# plt.legend()
plt.title('DAM RNN_168 Residual Errors - Train', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_168_train_residuals.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(dam_RNN_test_residuals,
         label='test_residuals',
         alpha = 0.9,
         color='darkorange')
# plt.legend()
plt.title('RNN_168 Residual Errors - Test', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_168_test_residuals.jpg', bbox_inches='tight')
plt.show();

# *** End of 7b_RNN_Model_DAM_20epochs_Lookbk168

#*** 100 Epochs
# RNN Model
model = Sequential()
    model.add(GRU(16,
                  input_shape=(batch_x.shape[1],
                               batch_x.shape[2]),
                               return_sequences=True))
                  model.add(GRU(16))
                  model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1)) # refers to nodes in the first hidden layer
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(lr=.0005),
              loss='mean_squared_error')
history = model.fit_generator(train_sequences,
                              validation_data=test_sequences,
                              epochs=100,
                              verbose=1)
'''Use tf.cast instead.
Epoch 1/100
341/341 [==============================] - 8s 24ms/step - loss: 1113.5844 - val_loss: 1869.4803
Epoch 2/100
341/341 [==============================] - 7s 20ms/step - loss: 428.6040 - val_loss: 1741.9301
Epoch 3/100
341/341 [==============================] - 7s 20ms/step - loss: 407.9612 - val_loss: 1679.8612
Epoch 4/100
341/341 [==============================] - 7s 20ms/step - loss: 355.9866 - val_loss: 1451.5491
Epoch 5/100
341/341 [==============================] - 7s 21ms/step - loss: 317.8952 - val_loss: 1312.4925
Epoch 6/100
341/341 [==============================] - 7s 20ms/step - loss: 285.2474 - val_loss: 1224.3213
Epoch 7/100
341/341 [==============================] - 7s 20ms/step - loss: 264.1806 - val_loss: 1145.9871
Epoch 8/100
341/341 [==============================] - 7s 21ms/step - loss: 247.0002 - val_loss: 1082.3395
Epoch 9/100
341/341 [==============================] - 7s 20ms/step - loss: 233.0637 - val_loss: 1043.8528
Epoch 10/100

341/341 [==============================] - 7s 20ms/step - loss: 220.5728 - val_loss: 1014.6569
Epoch 11/100
341/341 [==============================] - 7s 20ms/step - loss: 208.3335 - val_loss: 958.8172
Epoch 12/100
341/341 [==============================] - 7s 21ms/step - loss: 199.4702 - val_loss: 971.7406
Epoch 13/100
341/341 [==============================] - 7s 20ms/step - loss: 188.0252 - val_loss: 902.0690
Epoch 14/100
341/341 [==============================] - 7s 20ms/step - loss: 177.1703 - val_loss: 892.8538
Epoch 15/100
341/341 [==============================] - 7s 20ms/step - loss: 167.8670 - val_loss: 882.1721
Epoch 16/100
341/341 [==============================] - 7s 22ms/step - loss: 161.9555 - val_loss: 851.4034
Epoch 17/100
341/341 [==============================] - 7s 20ms/step - loss: 156.4996 - val_loss: 848.6315
Epoch 18/100
341/341 [==============================] - 7s 20ms/step - loss: 140.2908 - val_loss: 827.3252
Epoch 19/100
341/341 [==============================] - 7s 21ms/step - loss: 132.7221 - val_loss: 922.5107
Epoch 20/100
341/341 [==============================] - 7s 20ms/step - loss: 135.1146 - val_loss: 853.8203
Epoch 21/100
341/341 [==============================] - 7s 20ms/step - loss: 121.6766 - val_loss: 835.1993
Epoch 22/100
341/341 [==============================] - 7s 19ms/step - loss: 112.3490 - val_loss: 872.1540

Epoch 23/100
341/341 [==============================] - 7s 20ms/step - loss: 118.4664 - val_loss: 820.5470
Epoch 24/100
341/341 [==============================] - 7s 20ms/step - loss: 111.5561 - val_loss: 765.0522
Epoch 25/100
341/341 [==============================] - 7s 20ms/step - loss: 109.7488 - val_loss: 793.3327
Epoch 26/100
341/341 [==============================] - 7s 20ms/step - loss: 91.2481 - val_loss: 782.8136
Epoch 27/100
341/341 [==============================] - 7s 19ms/step - loss: 91.2702 - val_loss: 921.8957
Epoch 28/100
341/341 [==============================] - 6s 19ms/step - loss: 95.4983 - val_loss: 869.4399
Epoch 29/100
341/341 [==============================] - 7s 19ms/step - loss: 85.5055 - val_loss: 816.4147
Epoch 30/100
341/341 [==============================] - 7s 20ms/step - loss: 96.2177 - val_loss: 914.1672
Epoch 31/100
341/341 [==============================] - 7s 19ms/step - loss: 92.7546 - val_loss: 846.9522
Epoch 32/100
341/341 [==============================] - 7s 20ms/step - loss: 78.5323 - val_loss: 790.9805
Epoch 33/100
341/341 [==============================] - 7s 19ms/step - loss: 74.9790 - val_loss: 838.9070
Epoch 34/100
341/341 [==============================] - 6s 19ms/step - loss: 86.4126 - val_loss: 901.6393
Epoch 35/100

341/341 [==============================] - 7s 19ms/step - loss: 86.9925 - val_loss: 731.3899
Epoch 36/100
341/341 [==============================] - 7s 19ms/step - loss: 94.5935 - val_loss: 816.8768
Epoch 37/100
341/341 [==============================] - 7s 20ms/step - loss: 77.5772 - val_loss: 779.4698
Epoch 38/100
341/341 [==============================] - 7s 22ms/step - loss: 73.6988 - val_loss: 801.2102
Epoch 39/100
341/341 [==============================] - 7s 20ms/step - loss: 70.6784 - val_loss: 822.5007
Epoch 40/100
341/341 [==============================] - 7s 21ms/step - loss: 70.1964 - val_loss: 851.5249
Epoch 41/100
341/341 [==============================] - 7s 20ms/step - loss: 72.7054 - val_loss: 937.1877
Epoch 42/100
341/341 [==============================] - 7s 20ms/step - loss: 73.9277 - val_loss: 863.5355
Epoch 43/100
341/341 [==============================] - 7s 20ms/step - loss: 77.1547 - val_loss: 781.2635
Epoch 44/100
341/341 [==============================] - 7s 22ms/step - loss: 67.8980 - val_loss: 822.0585
Epoch 45/100
341/341 [==============================] - 7s 21ms/step - loss: 70.3691 - val_loss: 860.5074
Epoch 46/100
341/341 [==============================] - 6s 19ms/step - loss: 86.6551 - val_loss: 856.6004
Epoch 47/100
341/341 [==============================] - 7s 20ms/step - loss: 108.1198 - val_loss: 799.6595

Epoch 48/100
341/341 [==============================] - 8s 22ms/step - loss: 74.1508 - val_loss: 761.3187
Epoch 49/100
341/341 [==============================] - 7s 22ms/step - loss: 63.0470 - val_loss: 763.2052
Epoch 50/100
341/341 [==============================] - 7s 19ms/step - loss: 61.9480 - val_loss: 823.5684
Epoch 51/100
341/341 [==============================] - 7s 20ms/step - loss: 61.2882 - val_loss: 808.4469
Epoch 52/100
341/341 [==============================] - 8s 23ms/step - loss: 67.2412 - val_loss: 711.5163
Epoch 53/100
341/341 [==============================] - 8s 24ms/step - loss: 78.2741 - val_loss: 1090.5238
Epoch 54/100
341/341 [==============================] - 9s 27ms/step - loss: 78.3177 - val_loss: 802.7713
Epoch 55/100
341/341 [==============================] - 7s 21ms/step - loss: 67.8563 - val_loss: 709.6251
Epoch 56/100
341/341 [==============================] - 7s 22ms/step - loss: 72.5888 - val_loss: 867.1164
Epoch 57/100
341/341 [==============================] - 8s 23ms/step - loss: 64.2177 - val_loss: 764.0933
Epoch 58/100
341/341 [==============================] - 7s 21ms/step - loss: 70.3254 - val_loss: 805.6330
Epoch 59/100
341/341 [==============================] - 7s 20ms/step - loss: 63.8231 - val_loss: 847.1873
Epoch 60/100

341/341 [==============================] - 7s 22ms/step - loss: 70.6745 - val_loss: 876.8299
Epoch 61/100
341/341 [==============================] - 7s 20ms/step - loss: 61.2048 - val_loss: 856.0644
Epoch 62/100
341/341 [==============================] - 7s 21ms/step - loss: 62.0917 - val_loss: 790.6740
Epoch 63/100
341/341 [==============================] - 7s 19ms/step - loss: 63.6665 - val_loss: 862.9161
Epoch 64/100
341/341 [==============================] - 7s 19ms/step - loss: 61.3093 - val_loss: 845.2095
Epoch 65/100
341/341 [==============================] - 7s 19ms/step - loss: 61.7648 - val_loss: 853.5369
Epoch 66/100
341/341 [==============================] - 7s 19ms/step - loss: 59.7580 - val_loss: 881.3898
Epoch 67/100
341/341 [==============================] - 7s 20ms/step - loss: 76.3850 - val_loss: 768.6297
Epoch 68/100
341/341 [==============================] - 7s 21ms/step - loss: 72.7676 - val_loss: 770.1958
Epoch 69/100
341/341 [==============================] - 7s 21ms/step - loss: 57.4638 - val_loss: 816.6578
Epoch 70/100
341/341 [==============================] - 7s 19ms/step - loss: 55.0520 - val_loss: 782.5067
Epoch 71/100
341/341 [==============================] - 7s 19ms/step - loss: 60.4252 - val_loss: 824.6457
Epoch 72/100
341/341 [==============================] - 7s 19ms/step - loss: 63.4978 - val_loss: 778.6803

Epoch 73/100
341/341 [==============================] - 6s 19ms/step - loss: 55.9884 - val_loss: 786.2575
Epoch 74/100
341/341 [==============================] - 6s 19ms/step - loss: 57.3023 - val_loss: 805.5943
Epoch 75/100
341/341 [==============================] - 6s 19ms/step - loss: 69.8765 - val_loss: 811.4411
Epoch 76/100
341/341 [==============================] - 7s 19ms/step - loss: 60.7678 - val_loss: 767.6227
Epoch 77/100
341/341 [==============================] - 6s 19ms/step - loss: 60.2126 - val_loss: 833.9289
Epoch 78/100
341/341 [==============================] - 7s 20ms/step - loss: 53.8530 - val_loss: 755.3374
Epoch 79/100
341/341 [==============================] - 7s 20ms/step - loss: 54.6267 - val_loss: 908.6713
Epoch 80/100
341/341 [==============================] - 7s 19ms/step - loss: 64.6616 - val_loss: 751.0190
Epoch 81/100
341/341 [==============================] - 7s 21ms/step - loss: 60.3568 - val_loss: 822.1779
Epoch 82/100
341/341 [==============================] - 6s 19ms/step - loss: 60.9543 - val_loss: 797.5635
Epoch 83/100
341/341 [==============================] - 7s 20ms/step - loss: 59.0017 - val_loss: 795.2762
Epoch 84/100
341/341 [==============================] - 7s 19ms/step - loss: 52.3758 - val_loss: 841.4916
Epoch 85/100

341/341 [==============================] - 7s 19ms/step - loss: 58.1853 - val_loss: 753.5118
Epoch 86/100
341/341 [==============================] - 7s 21ms/step - loss: 54.7342 - val_loss: 833.8451
Epoch 87/100
341/341 [==============================] - 7s 20ms/step - loss: 59.8179 - val_loss: 730.4296
Epoch 88/100
341/341 [==============================] - 7s 21ms/step - loss: 54.6080 - val_loss: 748.5433
Epoch 89/100
341/341 [==============================] - 7s 21ms/step - loss: 58.5443 - val_loss: 764.8909
Epoch 90/100
341/341 [==============================] - 6s 18ms/step - loss: 74.7708 - val_loss: 813.5539
Epoch 91/100
341/341 [==============================] - 7s 19ms/step - loss: 56.0070 - val_loss: 734.7744
Epoch 92/100
341/341 [==============================] - 6s 19ms/step - loss: 48.7104 - val_loss: 732.2816
Epoch 93/100
341/341 [==============================] - 7s 20ms/step - loss: 53.9087 - val_loss: 762.7581
Epoch 94/100
341/341 [==============================] - 7s 22ms/step - loss: 47.9186 - val_loss: 815.8005
Epoch 95/100
341/341 [==============================] - 7s 20ms/step - loss: 53.9390 - val_loss: 761.0346
Epoch 96/100
341/341 [==============================] - 7s 20ms/step - loss: 60.3739 - val_loss: 777.2577
Epoch 97/100
341/341 [==============================] - 7s 20ms/step - loss: 54.2726 - val_loss: 811.3244

Epoch 98/100
341/341 [==============================] - 7s 20ms/step - loss: 49.2125 - val_loss: 782.5610
Epoch 99/100
341/341 [==============================] - 7s 20ms/step - loss: 51.3665 - val_loss: 746.4314
Epoch 100/100
341/341 [==============================] - 7s 20ms/step - loss: 57.4051 - val_loss: 762.8809
'''

train_loss = history.history['loss']
test_loss = history.history['val_loss']

plt.figure(figsize=(15, 8))
plt.plot(train_loss, label='Training MSE', color='darkblue')
plt.plot(test_loss, label='Testing MSE', color='darkorange')

plt.title('DAM RNN History ... 100 epochs, lookbk=12 hrs', fontsize=24)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss: Mean Squared Error', fontsize=18)
plt.legend()
plt.savefig('raw_data3/images/dam_RNN_12_100ep.jpg', bbox_inches='tight')
plt.show();

with open('raw_data3/fitted_models/dam_rnn_model_12_100ep.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('raw_data3/data/predictions/dam_rnn_history_12_100ep.pkl', 'wb') as f:
    pickle.dump(history, f)

# Evaluate

pred_train = []

y_train_true = []

for i in range(len(train_sequences)):
    pred_train.extend(model.predict(train_sequences[i][0]).ravel())
    y_train_true.extend(train_sequences[i][1].ravel())

print(f'Train pred: {len(pred_train)} rows')
print(f'Train y-true: {len(y_train_true)} rows')
# Train pred: 21788 rows
# Train y-true: 21788 rows

pred_test = []
y_test_true = []

for i in range(len(test_sequences)):
    pred_test.extend(model.predict(test_sequences[i][0]).ravel())
    y_test_true.extend(test_sequences[i][1].ravel())

print(f'Test pred: {len(pred_test)} rows')
print(f'Test y-true: {len(y_test_true)} rows')
# Test pred: 7255 rows
# Test y-true: 7255 rows

print(f'Train R2 score: {R_squared(y_train_true, pred_train):.4f}\
                         \nTest R2 score {R_squared(y_test_true, pred_test):.4f}\
                         \nOverfit: {R_squared(y_train_true, pred_train) - R_squared(y_test_true, pred_test):.4f}')
# Train R2 score: 0.8705
# Test R2 score 0.4717
# Overfit: 0.3988

dam_RNN_train_residuals = [y_train_true[i] - pred_train[i] for i in range(len(y_train_true))]
dam_RNN_test_residuals = [y_test_true[i] - pred_test[i] for i in range(len(y_test_true))]

plt.figure(figsize=(15,8))

plt.plot(y_train_true,
         label='train y-true',
         alpha = 0.8,
         color='darkblue')
plt.plot(pred_train,
         label='train predictions',
         alpha = 0.5,
         color='darkorange')
plt.legend()
plt.title('DAM - RNN_12_100ep Train: preds vs true (all)', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_12_100ep_train_pred_vs_true_all.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

# plt.plot(y_train[3:])
plt.plot(y_train_true,
         label='train y-true',
         alpha = 0.8,
         color='darkblue')
    plt.plot(pred_train,
             label='train predictions',
             alpha = 0.6,
             color='darkorange')
plt.legend()
plt.xlim(13_500, 14_750)
plt.title('DAM - RNN_12_100ep Train: preds vs true (zoomed2)', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_12_100ep_train_pred_vs_true_zoomed2.jpg',
bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(y_test_true,
         label='test y-true',
         alpha = 0.8,
         color='grey')
    plt.plot(pred_test,
             label='test predictions',
             alpha = 0.5,
             color='goldenrod')
plt.legend()
plt.xlim(600, 1200)

plt.title('DAM - RNN_12_100ep Test: preds vs true (zoomed)', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_12_100ep_test_pred_vs_true_zoomed.jpg',
            bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(dam_RNN_train_residuals,
label='train_residuals',
    alpha = 0.9,
    color='darkred')
# plt.legend()

plt.title('DAM RNN_12_100ep Residual Errors - Train', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_12_100ep_train_residuals.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(dam_RNN_test_residuals,
label='test_residuals',
    alpha = 0.9,
    color='darkorange')
# plt.legend()
plt.title('RNN_12_100ep Residual Errors - Test', fontsize=18)
plt.savefig('raw_data3/images/dam_RNN_12_100ep_test_residuals.jpg', bbox_inches='tight')

plt.show();

# End of 7c_RNN_Model_DAM_100epochs_Lookbk12

np.random.seed(2019)
# Load Pickles: Train/Test Dataframes & Scaled Arrays

with open('raw_data3/data/processed/train.pkl', 'rb') as f:
    train = pickle.load(f)

with open('raw_data3/data/processed/test.pkl', 'rb') as f:
    test = pickle.load(f)
# Create (scaled) Train-test X and y Pairs for Day Ahead RNN Model
features = [col for col in train.columns if col != 'hasp_price_per_mwh']
print(len(features))
print(features)
# 27
# ['dam_price_per_mwh', 'rtm_price_per_mwh', '7da_load_fcast_mw', '2da_load_fcast_mw',
'dam_load_fcast_mw', 'rtm_load_fcast_mw', 'water_acre_feet', 'sand_temp', 'sand_wind', 'sand_vis',
'sand_ceil', 'fres_temp', 'fres_wind', 'fres_vis', 'fres_ceil', 'rive_temp', 'rive_wind', 'rive_vis', 'rive_ceil',
'redd_temp', 'redd_wind', 'redd_vis', 'redd_ceil', 'year', 'month', 'day', 'hour']

X_train = train[features]
y_train = train['hasp_price_per_mwh']

X_test = test[features]
y_test = test['hasp_price_per_mwh']

X_train.shape
# (21800, 27)

ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

# Time Series Generator
# RNN Time Series Generator Parameters

lookback_length = 18
batch_size = 64

train_sequences = TimeseriesGenerator(X_train_sc,
                                      y_train,
                                      length=lookback_length,
                                      batch_size=batch_size)
    batch_x, batch_y = train_sequences[0]

test_sequences = TimeseriesGenerator(X_test_sc,
                                     y_test,
                                     length=lookback_length,
                                     batch_size=batch_size)

len(train_sequences)

# 341

train_sequences[0][0].shape
# (64, 18, 27)

print(batch_x.shape)
print(batch_y.shape)
# (64, 18, 27)
# (64,)

# RNN Model
np.random.seed(2019)
model = Sequential()
model.add(GRU(32,
              input_shape=(batch_x.shape[1],
                           batch_x.shape[2]),
                           return_sequences=True))
              model.add(GRU(16))
              model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1)) # refers to nodes in the first hidden layer
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(lr=.0001),
loss='mean_squared_error')
history = model.fit_generator(train_sequences,
                              validation_data=test_sequences,
                              epochs=15,
                              verbose=1)
'''
Epoch 1/15
341/341 [==============================] - 11s 32ms/step - loss: 2708.2206 - val_loss: 2988.4103
Epoch 2/15
341/341 [==============================] - 10s 28ms/step - loss: 2430.3186 - val_loss: 2156.4228
Epoch 3/15
341/341 [==============================] - 10s 29ms/step - loss: 1777.8309 - val_loss: 1288.2555
Epoch 4/15
341/341 [==============================] - 10s 30ms/step - loss: 1610.5455 - val_loss: 1114.7500
Epoch 5/15
341/341 [==============================] - 10s 29ms/step - loss: 1570.4313 - val_loss: 1037.2566
Epoch 6/15
341/341 [==============================] - 11s 32ms/step - loss: 1548.3631 - val_loss: 992.3690
Epoch 7/15
341/341 [==============================] - 11s 32ms/step - loss: 1529.0917 - val_loss: 958.1654
Epoch 8/15
341/341 [==============================] - 11s 32ms/step - loss: 1509.6074 - val_loss: 926.9646
Epoch 9/15

341/341 [==============================] - 12s 34ms/step - loss: 1491.4554 - val_loss: 879.5915
Epoch 10/15
341/341 [==============================] - 12s 34ms/step - loss: 1472.8037 - val_loss: 849.8519
Epoch 11/15
341/341 [==============================] - 11s 33ms/step - loss: 1454.6789 - val_loss: 820.0376
Epoch 12/15
341/341 [==============================] - 11s 31ms/step - loss: 1436.5245 - val_loss: 798.7180
Epoch 13/15
341/341 [==============================] - 10s 30ms/step - loss: 1419.7284 - val_loss: 776.6583
Epoch 14/15
341/341 [==============================] - 10s 28ms/step - loss: 1403.0605 - val_loss: 755.4619
Epoch 15/15
341/341 [==============================] - 11s 33ms/step - loss: 1386.8152 - val_loss: 739.9310
'''

train_loss = history.history['loss']
test_loss = history.history['val_loss']

plt.figure(figsize=(15, 8))
plt.plot(train_loss, label='Training loss', color='darkblue')
plt.plot(test_loss, label='Testing loss', color='darkorange')
plt.savefig('raw_data3/images/hasp_rnn_18_epochs.jpg', bbox_inches='tight')
plt.legend();

with open('raw_data3/fitted_models/hasp_rnn_model_18.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('raw_data3/data/predictions/hasp_rnn_history_18.pkl', 'wb') as f:
    pickle.dump(history, f)
# Evaluate
pred_train = []
y_train_true = []

for i in range(len(train_sequences)):
    pred_train.extend(model.predict(train_sequences[i][0]).ravel())
    y_train_true.extend(train_sequences[i][1].ravel())

print(f'Train pred: {len(pred_train)} rows')
print(f'Train y-true: {len(y_train_true)} rows')
# Train pred: 21782 rows
# Train y-true: 21782 rows

pred_test = []
y_test_true = []

for i in range(len(test_sequences)):
    pred_test.extend(model.predict(test_sequences[i][0]).ravel())
    y_test_true.extend(test_sequences[i][1].ravel())

print(f'Test pred: {len(pred_test)} rows')
print(f'Test y-true: {len(y_test_true)} rows')
# Test pred: 7249 rows
# Test y-true: 7249 rows

print(f'Train R2 score: {R_squared(y_train_true, pred_train):.4f}\
                         \nTest R2 score {R_squared(y_test_true, pred_test):.4f}\
                         \nOverfit: {R_squared(y_train_true, pred_train) - R_squared(y_test_true, pred_test):.4f}')
# Train R2 score: 0.1804
# Test R2 score 0.3395
# Overfit: -0.1591

dam_RNN_train_residuals = [y_train_true[i] - pred_train[i] for i in range(len(y_train_true))]
dam_RNN_test_residuals = [y_test_true[i] - pred_test[i] for i in range(len(y_test_true))]

plt.figure(figsize=(15,8))
plt.plot(y_train_true,
         label='train y-true',
         alpha = 0.8,
         color='darkblue')
    plt.plot(pred_train,
             label='train predictions',
             alpha = 0.5,
             color='darkorange')
plt.legend()
plt.title('HASP - RNN_18 Train: preds vs true (all)', fontsize=18)
plt.savefig('raw_data3/images/hasp_RNN_18_train_pred_vs_true_all.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(y_train_true,
         label='train y-true',
         alpha = 0.8,
         color='darkblue')
plt.plot(pred_train,
         label='train predictions',
         alpha = 0.6,
         color='darkorange')
plt.legend()
plt.xlim(13_500, 14_750)
plt.title('HASP - RNN_18 Train: preds vs true (zoomed2)', fontsize=18)
plt.savefig('raw_data3/images/hasp_RNN_18_train_pred_vs_true_zoomed2.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(y_test_true,
         label='test y-true',
         alpha = 0.8,
         color='grey')
plt.plot(pred_test,
         label='test predictions',
         alpha = 0.5,
         color='goldenrod')
plt.legend()
plt.xlim(600, 1200)

plt.title('HASP - RNN_18 Test: preds vs true (zoomed)', fontsize=18)
plt.savefig('raw_data3/images/hasp_RNN_18_test_pred_vs_true_zoomed.jpg', bbox_inches='tight')
plt.show();

plt.figure(figsize=(15,8))

plt.plot(dam_RNN_train_residuals,
         label='train_residuals',
         alpha = 0.9,
         color='darkred')
# plt.legend()
plt.title('HASP RNN_18 Residual Errors - Train', fontsize=18)
plt.savefig('raw_data3/images/hasp_RNN_18_train_residuals.jpg', bbox_inches='tight')

plt.show();

plt.figure(figsize=(15,8))

plt.plot(dam_RNN_test_residuals,
         label='test_residuals',
         alpha = 0.9,
         color='darkorange')
# plt.legend()
plt.title('HASP RNN_18 Residual Errors - Test', fontsize=18)
plt.savefig('raw_data3/images/hasp_RNN_18_test_residuals.jpg', bbox_inches='tight')
plt.show();

# *** End of 8a_RNN_Model_Hour-Ahead_15epochs_Lookbk18