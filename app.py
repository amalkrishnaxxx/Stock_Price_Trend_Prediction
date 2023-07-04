import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from yahoofinancials import YahooFinancials
from keras.models import load_model
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


start='2013-1-1'
end='2022-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'TSLA')
df = yf.download(user_input, start, end, progress=False)

#describing data
st.subheader('Data from 2013 - 2022')
st.write(df.describe())

#visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close, label='Closing Price')
plt.legend()
st.pyplot(fig)

#ma = moving average

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100, label='MA100')
plt.plot(df.Close, label='Closing Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100, label='MA100')
plt.plot(ma200, label='MA200')
plt.plot(df.Close, label='Closing Price')
plt.legend()
st.pyplot(fig)


df_training = df['Close'][0:int(len(df)*0.70)]
df_testing = df['Close'][int(len(df)*0.70): int(len(df))]
df_training = pd.DataFrame(df_training, columns=['Close'])
df_testing = pd.DataFrame(df_testing, columns=['Close'])



scaler = MinMaxScaler(feature_range = (0,1))

df_training_array = scaler.fit_transform(df_training)


#Loading the model

model = load_model('keras_model.h5')

#Testing 

past_100_days =df_training.tail(100)
final_df = pd.concat([past_100_days, df_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(pd.DataFrame(input_data[i-100:i,:]))
    y_test.append(pd.DataFrame([input_data[i, 0]]))



x_test = np.array(x_test)
y_test = np.array(y_test).reshape(-1,1)

y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Prediction visualization
st.subheader('Predicted vs Original Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
