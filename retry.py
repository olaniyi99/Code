import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

parse_dates = ['Date']

load_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/hourlyload_data.csv')#half hourly power demand data 
met_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/fullyear_temp_data_london.csv') #meterological hourly data
london_mobility_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/London_mobility.csv') #daily mobility data

load_data['Date'] = pd.to_datetime(load_data['Date'] + ' '  + load_data['Time'])
met_data['Date'] = pd.to_datetime(met_data['Date'] + ' '  + met_data['Time'])


dates = load_data.Date.values

load_data.drop('Time', inplace=True, axis=1)
met_data.drop('Time', inplace=True, axis=1)

temp_original = met_data.Temp.values
humidity_original = met_data.Humidity.values
dew_original = met_data.DewPoint.values
mobility_original = london_mobility_data.Index.values
D = load_data.Demand.values #half hourly power demand values 
date = load_data.Date.values

dew = np.zeros(len(D))
temp = np.zeros(len(D))
humidity = np.zeros(len(D))
mobility = np.zeros(len(D))

for i in range(len(D)):
    n_day = int(i/24)
    temp[i] = temp_original[n_day]
    dew[i] = dew_original[n_day] 
    humidity[i] = humidity_original[n_day]
    mobility[i] = mobility_original[n_day]

test_data = {
            'Demand': D,
            'Temp': temp,
            'Dew': dew,
            'Humidity': humidity,
            'Mobility': mobility
            }

test_data = pd.DataFrame(test_data)
scaler = MinMaxScaler()

test_data_1 =scaler.fit_transform(test_data)
print(test_data_1.shape)

features = test_data_1
print(features.shape)
target = test_data_1[:,0]
print(target.shape)


#features = test_data[['Demand', 'Temp', 'Dew', 'Humidity']].to_numpy().tolist()
#target  = test_data['Demand'].tolist()

ts_generator = TimeseriesGenerator(features, target, length=24, sampling_rate=1, batch_size=1, stride=1)


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123, shuffle = False)

win_length = 9
num_features = 5

train_generator = TimeseriesGenerator(x_train, y_train, length = win_length, sampling_rate=1, batch_size=32)
test_generator = TimeseriesGenerator(x_test, y_test, length=win_length, sampling_rate=1, batch_size=32)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(win_length, num_features), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

model.summary()

#early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=2, mode='min')
model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
#callbacks=[early_stopping]
history = model.fit_generator(train_generator, epochs=10, validation_data = test_generator, shuffle=False )

model.evaluate_generator(test_generator, verbose=0)

predictions = model.predict_generator(ts_generator)
print(predictions.shape)

df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:,1:][win_length:])], axis=1)

rev_trans = scaler.inverse_transform(df_pred)


df_final = test_data[predictions.shape[0]*-1:]
df_final["Demand_pred"] = rev_trans[:,0]


plt.plot(df_final['Demand'], color = 'black', label = 'Actual Load Demand')
plt.plot(df_final['Demand_pred'], color = 'green', label = 'Predicted Load Demand')
plt.title('Load Demand Prediction')
#plt.xlabel('Date')
plt.ylabel('Power')
plt.legend()
plt.show()



#df_final[['Demand', 'Demand_pred']].plot(x=dates)
plt.show()

#load_data.set_index('Date')['Demand'].plot()
#plt.show()

