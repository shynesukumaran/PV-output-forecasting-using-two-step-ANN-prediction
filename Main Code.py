#Load libraris required 

import pandas as pd
import numpy as np
import glob
import math
import pvlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from math import sqrt
from numpy import split
import math
import requests
import urllib3
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import datetime as dt
import html5lib
import requests
import openpyxl
import json
import time, threading
import tensorflow as tf
import tensorflow.keras 
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input, Concatenate,Flatten,RepeatVector,TimeDistributed

import keras
import tensorflow as tf
tf.Session(config=tf.ConfigProto(device_count = {'GPU': 1}))


path = r'C:\Users\sukumara\Downloads\Thesis files\Data_from_plant' # use your path
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)  
frame = pd.concat(li, axis=1, ignore_index=True)
print(frame)


frame =frame.rename(columns={0 : 'Date and Time',1:'Active Power',2:'D1',3:'Air Temperature',4:'D2',5:'Radiation',6:'D3',7:'Wind Speed'}
finaldata = frame.drop(columns=['D1', 'D2','D3'])
                    
data1 = finaldata
values1 = data1['Active Power'].values
values2 = data1['Air Temperature'].values
values3 = data1['Radiation'].values
values4 = data1['Wind Speed'].values

values1 = values1.reshape((len(values1), 1))
values2 = values2.reshape((len(values2), 1))
values3 = values3.reshape((len(values3), 1))
values4 = values4.reshape((len(values4), 1))

# train the normalization
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler3 = MinMaxScaler(feature_range=(0, 1))
scaler4 = MinMaxScaler(feature_range=(0, 1))

scaler1 = scaler1.fit(values1)
scaler2 = scaler2.fit(values2)
scaler3 = scaler3.fit(values3)
scaler4 = scaler4.fit(values4)

norm1 = scaler1.transform(values1)
norm2 = scaler2.transform(values2)
norm3 = scaler3.transform(values3)
norm4 = scaler4.transform(values4)


norm1=norm1.flatten()
norm2=norm2.flatten()
norm3=norm3.flatten()
norm4=norm4.flatten()


series1 = pd.Series(norm1)
series2 = pd.Series(norm2)
series3 = pd.Series(norm3)
series4 = pd.Series(norm4)
                    
                    
data2 = pd.concat([finaldata["Date and Time"], series1,series2,series3,series4], axis=1)
data2.columns = ['Date and Time', 'Active Power','Air_temperature','Radiation','Wind Speed']
train_size = int(len(data2)*0.80)
train, test = data2[0:train_size], data2[train_size:len(data2)]
train = train.replace(np.inf, np.nan)
test = test.replace(np.inf, np.nan)
train = train.fillna(0)
test = test.fillna(0)                    
print(train)
print(test)

X_train = train.iloc[:,2:5]
y_train = train['Active Power'].values
X_test = test.iloc[:,2:5]
y_test = test['Active Power'].values
print(X_train)
print(y_train)
print(X_test)
print(y_test)
                    
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=3, activation = 'relu'))
nn_model.add(Dense(16, activation = 'relu'))
nn_model.add(Dense(8, activation = 'relu'))
nn_model.add(Dense(1))
nn_model.summary()
                    
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

nn_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history = nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, callbacks=[early_stop], shuffle=False)
                    
y_pred_test_nn = nn_model.predict(X_test)
y_pred_train_nn = nn_model.predict(X_train)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train_nn)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))                   
                    

nn_model.save(r"C:\Users\sukumara\Downloads\Thesis files\Models\model_for_pv_pred.h5")
print("Saved model to disk")
                    
                    
                    

frame = pd.read_csv(r"C:\Users\sukumara\Desktop\All files\PV Project\Solar_radiation_training_data.csv",error_bad_lines=False,sep=';',decimal=',')
#frame = frame.drop(columns={'Unnamed: 0'})
frame

data3 = frame
values11 = data3['Wind_speed'].values
values12 = data3['Air_temperature'].values
values13 = data3['Relative_humidity'].values
values14 = data3['Air_pressure'].values
values15 = data3['Cloud_cover'].values
values16 = data3['Radiation'].values

values11 = values11.reshape((len(values11), 1))
values12 = values12.reshape((len(values12), 1))
values13 = values13.reshape((len(values13), 1))
values14 = values14.reshape((len(values14), 1))
values15 = values15.reshape((len(values15), 1))
values16 = values16.reshape((len(values16), 1))


scaler11 = MinMaxScaler(feature_range=(0, 1))
scaler12 = MinMaxScaler(feature_range=(0, 1))
scaler13 = MinMaxScaler(feature_range=(0, 1))
scaler14 = MinMaxScaler(feature_range=(0, 1))
scaler15 = MinMaxScaler(feature_range=(0, 1))
scaler16 = MinMaxScaler(feature_range=(0, 1))

scaler11 = scaler11.fit(values11)
scaler12 = scaler12.fit(values12)
scaler13 = scaler13.fit(values13)
scaler14 = scaler14.fit(values14)
scaler15 = scaler15.fit(values15)
scaler16 = scaler16.fit(values16)

norm11 = scaler11.transform(values11)
norm12 = scaler12.transform(values12)
norm13 = scaler13.transform(values13)
norm14 = scaler14.transform(values14)
norm15 = scaler15.transform(values15)
norm16 = scaler16.transform(values16)

norm11=norm11.flatten()
norm12=norm12.flatten()
norm13=norm13.flatten()
norm14=norm14.flatten()
norm15=norm15.flatten()
norm16=norm16.flatten()


series11 = pd.Series(norm11)
series12 = pd.Series(norm12)
series13 = pd.Series(norm13)
series14 = pd.Series(norm14)
series15 = pd.Series(norm15)
series16 = pd.Series(norm16)                    
                    
data4 = pd.concat([frame["Date and Time"], series11,series12,series13,series14,series15,series16], axis=1)
data4.columns = ['Date and Time','Wind_speed','Air_temperature','Relative_humidity','Air_pressure','Cloud_cover','Radiation']
data4
                    
train_size = int(len(data4)*0.65)
train, test = data4[0:train_size], data4[train_size:len(data4)]
train = train.replace(np.inf, np.nan)
test = test.replace(np.inf, np.nan)
train = train.fillna(0)
test = test.fillna(0)                   
print(train)
print(test)
              
                    
X_train = train.iloc[:,1:6]
y_train = train['Radiation'].values
X_test = test.iloc[:,1:6]
y_test = test['Radiation'].values
print(X_train)
print(y_train)
print(X_test)
print(y_test)
                    
nn_model_rad = Sequential()
nn_model_rad.add(Dense(65, input_dim=5, activation = 'relu'))
nn_model_rad.add(Dense(13, activation = 'relu'))
nn_model_rad.add(Dense(5, activation = 'relu'))
nn_model_rad.add(Dense(1))
nn_model_rad.summary()
                 
nn_model_rad.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=100, verbose=1)
history = nn_model_rad.fit(X_train, y_train, epochs=500, batch_size=600, verbose=1, callbacks=[early_stop], shuffle=False)
                    
                    
y_pred_test_nn = nn_model_rad.predict(X_test)
y_pred_train_nn = nn_model_rad.predict(X_train)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train_nn)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))
                    
test_plot=pd.DataFrame()
test_plot['Actual']=y_test
test_plot['Predicted']=y_pred_test_nn
test_plot['Time']=test['Date and Time'].values
test_plot
                    
plt.figure(figsize=(25,10))


fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(25,18),sharex=True, sharey=True)
ax = plt.gca()



fig.suptitle('Comparison: Actual vs Predicted Solar radiation for the test set\n\n R2 Score=0.62', fontsize=20)
test_plot.plot(kind='line',x='Time', y='Predicted',color = '#3EF70C', ax=axes[0])
test_plot.plot(kind='line',x='Time',y='Actual', color='#FC0404', ax=axes[1])
fig.text(0.04,0.6,'Raidation- GHI (normalised values)', va='center', rotation='vertical',fontsize=16)
plt.gcf().autofmt_xdate()
plt.savefig('Final_Prediction_Comparison.png', dpi = 300)
plt.show()
                    
                    
response = requests.get("*API CALL LINK")
print(response.status_code) 
                    
data = response.json()
print(data)
                    
                    
test_weather = pd.DataFrame(dtype = float)
header = ["Date and Time","Wind_speed","Air_temperature","Relative_humidity","Air_pressure","Cloud_cover"]
temperature = []
pressure = []
humidity = []
wind_speed = []
cloud_cover=[]
date = []
for i in range(len(data.get('hourly'))):    
    temperature.append(data.get('hourly')[i].get('temp'))
    pressure.append(data.get('hourly')[i].get('pressure'))
    humidity.append(data.get('hourly')[i].get('humidity'))
    wind_speed.append(data.get('hourly')[i].get('wind_speed'))
    cloud_cover.append(data.get('hourly')[i].get('clouds'))
    timestamp=data.get('hourly')[i].get('dt')
    date_time=datetime.fromtimestamp(timestamp)
    date.append(date_time) 
    
test_weather[header[0]] = date
test_weather[header[1]] = wind_speed
test_weather[header[2]] = temperature
test_weather[header[3]] = humidity
test_weather[header[4]] = pressure
test_weather[header[5]] = cloud_cover

test_weather['Cloud_cover']=test_weather['Cloud_cover']*(0.01)                        
print(test_weather)                    
                    
                    
                    
data1 = test_weather
values5 = data1['Wind_speed'].values
values6 = data1['Air_temperature'].values
values7 = data1['Relative_humidity'].values
values8 = data1['Air_pressure'].values
values9 = data1['Cloud_cover'].values

values5 = values5.reshape((len(values5), 1))
values6 = values6.reshape((len(values6), 1))
values7 = values7.reshape((len(values7), 1))
values8 = values8.reshape((len(values8), 1))
values9 = values9.reshape((len(values9), 1))

scaler5 = MinMaxScaler(feature_range=(0, 1))
scaler6 = MinMaxScaler(feature_range=(0, 1))
scaler7 = MinMaxScaler(feature_range=(0, 1))
scaler8 = MinMaxScaler(feature_range=(0, 1))
scaler9 = MinMaxScaler(feature_range=(0, 1))

scaler5 = scaler5.fit(values5)
scaler6 = scaler6.fit(values6)
scaler7 = scaler7.fit(values7)
scaler8 = scaler8.fit(values8)
scaler9 = scaler9.fit(values9)

norm5 = scaler5.transform(values5)
norm6 = scaler6.transform(values6)
norm7 = scaler7.transform(values7)
norm8 = scaler8.transform(values8)
norm9 = scaler9.transform(values9)

norm5=norm5.flatten()
norm6=norm6.flatten()
norm7=norm7.flatten()
norm8=norm8.flatten()
norm9=norm9.flatten()

series5 = pd.Series(norm5)
series6 = pd.Series(norm6)
series7 = pd.Series(norm7)
series8 = pd.Series(norm8)
series9 = pd.Series(norm9)                    


weather_forecast= pd.concat([test_weather["Date and Time"], series5,series6,series7,series8,series9], axis=1)
weather_forecast.columns = ['Date and Time', 'Wind_speed','Air_temperature','Relative_humidity','Air_pressure','Cloud_cover']
weather_forecast                    
                    
X_test = weather_forecast.iloc[:,1:6]
X_test

solar_rad_pred = nn_model_rad.predict(X_test)
solar_rad_pred = solar_rad_pred.flatten()
series10 = pd.Series(solar_rad_pred)
series10

                    
forecast_pv= pd.concat([weather_forecast["Date and Time"], series6,series10,series5], axis=1)
forecast_pv.columns = ['Date and Time','Air_temperature','Radiation','Wind_speed']
forecast_pv
                
X_test2 = forecast_pv.iloc[:,1:4]
X_test2
             
pv_prediction = nn_model.predict(X_test2)
pv_prediction
                  
pv_pred = scaler1.inverse_transform(pv_prediction)
pv_pred
               
pv_pred=pv_pred.flatten()
series_pv = pd.Series(pv_pred)
series_pv
                    
final_pv_pred= pd.concat([weather_forecast["Date and Time"], series_pv], axis=1)
final_pv_pred.columns = ['Date and Time','Predicted PV_Power']
final_pv_pred
                    
plt.figure(figsize=(15,8))
ax = plt.gca()
final_pv_pred.plot(kind='line',x='Date and Time',y='Predicted PV_Power',color = 'green', ax=ax)
plt.title('Prediction of PV output power for the next 48 hours', fontsize = 12)
plt.ylabel('Active Power in kilowatts (kW)', fontsize = 12)
plt.xlabel('Date and Time', fontsize = 12)
plt.gcf().autofmt_xdate()
plt.savefig('active_power_output.png', dpi = 300)
plt.show()                
