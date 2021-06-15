
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
parse_dates = ['Date']
load_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/Fullyeardata.csv', parse_dates=parse_dates, index_col='Date') #half hourly power demand data 
met_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/fullyear_temp_data_london.csv', parse_dates=parse_dates, index_col='Date') #meterological hourly data
london_mobility_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/London_mobility.csv', parse_dates=parse_dates, index_col='Date') #daily mobility data


#max and min for load data
load_year_max = max(load_data['Demand']) #max power demand for the year 
load_year_min = min(load_data['Demand']) #min power demand for the year 
D_max_daily = load_data.groupby('Date').Demand.max().values #daily maximum demand
D_min_daily = load_data.groupby('Date').Demand.min().values #daily min demand
D = load_data.Demand.values #half hourly power demand values 


#max and min for met data
temp_year_max = max(met_data['Temp'])
temp_year_min = min(met_data['Temp'])
temp_max_daily = met_data.groupby('Date').Temp.max().values
temp_min_daily = met_data.groupby('Date').Temp.min().values
temp_original = met_data.Temp.values

dew_year_max = max(met_data['DewPoint'])
dew_year_min = min(met_data['DewPoint'])
dew_max_daily = met_data.groupby('Date').DewPoint.max().values
dew_min_daily = met_data.groupby('Date').DewPoint.min().values
dew_original = met_data.DewPoint.values

humidity_year_max = max(met_data['Humidity'])
humidity_year_min = min(met_data['Humidity'])
humidity_max_daily = met_data.groupby('Date').Humidity.max().values
humidity_min_daily = met_data.groupby('Date').Humidity.min().values
humidity_original = met_data.Humidity.values

precip_year_max = max(met_data['Precipitation'])
precip_year_min = min(met_data['Precipitation'])
precip_max_daily = met_data.groupby('Date').Precipitation.max().values
precip_min_daily = met_data.groupby('Date').Precipitation.min().values
precip_original = met_data.Precipitation.values

wdir_year_max = max(met_data['WindDirection'])
wdir_year_min = min(met_data['WindDirection'])
wdir_max_daily = met_data.groupby('Date').WindDirection.max().values
wdir_min_daily = met_data.groupby('Date').WindDirection.min().values
wdir_original = met_data.WindDirection.values

wspeed_year_max = max(met_data['WindSpeed'])
wspeed_year_min = min(met_data['WindSpeed'])
wspeed_max_daily = met_data.groupby('Date').WindSpeed.max().values
wspeed_min_daily = met_data.groupby('Date').WindSpeed.min().values
wspeed_original = met_data.WindSpeed.values

wgust_year_max = max(met_data['WindGust'])
wgust_year_min = min(met_data['WindGust'])
wgust_max_daily = met_data.groupby('Date').WindGust.max().values
wgust_min_daily = met_data.groupby('Date').WindGust.min().values
wgust_original = met_data.WindGust.values

pressure_year_max = max(met_data['Pressure'])
pressure_year_min = min(met_data['Pressure'])
pressure_max_daily = met_data.groupby('Date').Pressure.max().values
pressure_min_daily = met_data.groupby('Date').Pressure.min().values
pressure_original = met_data.Pressure.values

#max and min for london mobility data
#daily max and min not included since dataset is already done on a daily basis rather than hourly/half hourly
mobility_year_max = max(london_mobility_data['Index'])
mobility_year_min = min(london_mobility_data['Index'])
mobility_original = london_mobility_data.Index.values

#initialise array for max and min values
#make arrays the same size as the array with power demand values
D_max = np.zeros(len(D))
D_min = np.zeros(len(D))

temp_max = np.zeros(len(D))
temp_min = np.zeros(len(D))
temp = np.zeros(len(D))

dew_max = np.zeros(len(D))
dew_min= np.zeros(len(D))
dew = np.zeros(len(D))

humidity_max = np.zeros(len(D))
humidity_min = np.zeros(len(D))
humidity = np.zeros(len(D))

precip_max = np.zeros(len(D))
precip_min = np.zeros(len(D))
precip = np.zeros(len(D))

wdir_max = np.zeros(len(D))
wdir_min = np.zeros(len(D))
wdir = np.zeros(len(D))

wspeed_max = np.zeros(len(D))
wspeed_min = np.zeros(len(D))
wspeed = np.zeros(len(D))

wgust_max = np.zeros(len(D))
wgust_min = np.zeros(len(D))
wgust = np.zeros(len(D))

pressure_max = np.zeros(len(D))
pressure_min = np.zeros(len(D))
pressure = np.zeros(len(D))

mobility = np.zeros(len(D))

#link the max value for each day to the half hourly load forecasts 
#for example, if max demand on 01/01/2001 = 20, then make sure at 00:00 up to 23:30 on 01/01/2001 all have max demand = 20
for i in range(len(D)):
    n_day = int(i/46)
    D_max[i] = D_max_daily[n_day]
    D_min[i] = D_min_daily[n_day]


    temp_max[i] = temp_max_daily[n_day]
    temp_min[i] = temp_min_daily[n_day]  
    temp[i] = temp_original[n_day]

    dew_max[i] = dew_max_daily[n_day]
    dew_min[i] = dew_min_daily[n_day]  
    dew[i] = dew_original[n_day] 

    humidity_max[i] = humidity_max_daily[n_day]
    humidity_min[i] = humidity_min_daily[n_day]
    humidity[i] = humidity_original[n_day]

    precip_max[i] = precip_max_daily[n_day]
    precip_min[i] = precip_min_daily[n_day]
    precip[i] = precip_original[n_day]

    wdir_max[i] = wdir_max_daily[n_day]
    wdir_min[i] = wdir_min_daily[n_day]
    wdir[i] = wdir_original[n_day]

    wspeed_max[i] = wspeed_max_daily[n_day]
    wspeed_min[i] = wspeed_min_daily[n_day]
    wspeed[i] = wspeed_original[n_day]

    wgust_max[i] = wgust_max_daily[n_day]
    wgust_min[i] = wgust_min_daily[n_day]
    wgust[i] = wgust_original[n_day]

    pressure_max[i] = pressure_max_daily[n_day]
    pressure_min[i] = pressure_min_daily[n_day]
    pressure[i] = pressure_original[n_day]

    mobility[i] = mobility_original[n_day] 

#normalisation of daily max, min and demand data using scaling to range
D = (D - load_year_min)/(load_year_max - load_year_min)
D_max = (D_max - load_year_min)/(load_year_max - load_year_min)
D_min = (D_min - load_year_min)/(load_year_max - load_year_min)

temp = (temp - temp_year_min)/(temp_year_max - temp_year_min)
temp_max = (temp_max - temp_year_min)/(temp_year_max - temp_year_min)
temp_min = (temp_min - temp_year_min)/(temp_year_max - temp_year_min)

dew = (dew - dew_year_min)/(dew_year_max - dew_year_min)
dew_max = (dew_max - dew_year_min)/(dew_year_max - dew_year_min)
dew_min = (dew_min - dew_year_min)/(dew_year_max - dew_year_min)

humidity = (humidity - humidity_year_min)/(humidity_year_max - humidity_year_min)
humidity_max = (humidity_max - humidity_year_min)/(humidity_year_max - humidity_year_min)
humidity_min = (humidity_min - humidity_year_min)/(humidity_year_max - humidity_year_min)

precip = (precip - precip_year_min)/(precip_year_max - precip_year_min)
precip_max = (precip_max - precip_year_min)/(precip_year_max - precip_year_min)
precip_min = (precip_min - precip_year_min)/(precip_year_max - precip_year_min)

wdir = (wdir - wdir_year_min)/(wdir_year_max - wdir_year_min)
wdir_max = (wdir_max - wdir_year_min)/(wdir_year_max - wdir_year_min)
wdir_min = (wdir_min - wdir_year_min)/(wdir_year_max - wdir_year_min)

wspeed = (wspeed - wspeed_year_min)/(wspeed_year_max - wspeed_year_min)
wspeed_max = (wspeed_max - wspeed_year_min)/(wspeed_year_max - wspeed_year_min)
wspeed_min = (wdir_min - wdir_year_min)/(wdir_year_max - wdir_year_min)

wgust = (wgust - wgust_year_min)/(wgust_year_max - wgust_year_min)
wgust_max = (wgust_max - wgust_year_min)/(wgust_year_max - wgust_year_min)
wgust_min = (wgust_min - wgust_year_min)/(wgust_year_max - wgust_year_min)

pressure = (pressure - pressure_year_min)/(pressure_year_max - pressure_year_min)
pressure_max = (pressure_max - pressure_year_min)/(pressure_year_max - pressure_year_min)
pressure_min = (pressure_min - pressure_year_min)/(pressure_year_max - pressure_year_min)

mobility = (mobility - mobility_year_min)/(mobility_year_max - mobility_year_min) 


data_f = [D, D_max, D_min, temp, temp_max, temp_min, dew, dew_max, dew_min, humidity, humidity_max, humidity_min, precip, precip_max, precip_min, wdir, wdir_max, wdir_min, wspeed, wspeed_max, wspeed_min, wgust, wgust_max, wgust_min, pressure, pressure_max, pressure_min]
data_final = pd.DataFrame(data_f)

data_final.to_csv('df_train_defl.csv', index=False)


#data split, need 24 subnetworks to forecast load for each hour
#from sklearn.model_selection import train_test_split


#from keras.models import Sequential
#from keras.layers import Dense