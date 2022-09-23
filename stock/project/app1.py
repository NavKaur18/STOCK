# # Importing flask module in the project is mandatory
# # An object of Flask class is our WSGI application.
# from flask import Flask
 
# # Flask constructor takes the name of
# # current module (__name__) as argument.
# app = Flask(__name__)
 
# # The route() function of the Flask class is a decorator,
# # which tells the application which URL should call
# # the associated function.
# @app.route('/')
# # ‘/’ URL is bound with hello_world() function.
# def hello_world():
#     return 'Hello World'
 
# # main driver function
# if __name__ == '__main__':
 
#     # run() method of Flask class runs the application
#     # on the local development server.
#     app.run()

from pyexpat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
from keras.models import load_model

start = '2010-01-01'
end ='2022-06-01'


st. markdown("<h1 style='text-align: center; color: Green;'>Stock Trend Prediction</h1>", unsafe_allow_html=True)


user_input=st.text_input('Enter Stock Ticker','AAPL')
df= data.DataReader('AAPL','yahoo',start, end)
df= data.DataReader('TSLA','yahoo',start, end)

#Describing the Data
st.subheader('Data from 2010-2022')
st.write(df.describe())

#Visualization
st. markdown("<h1 style='text-align: center; color: White;'>Closing Price vd Time chart</h1>", unsafe_allow_html=True)

fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st. markdown("<h1 style='text-align: center; color: White;'>Closing Price vd Time chart with 100(Moving Average)</h1>", unsafe_allow_html=True)

ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st. markdown("<h1 style='text-align: center; color: White;'>Closing Price vd Time Chart with 100(Moving Average) & 200(Moving Average)</h1>", unsafe_allow_html=True)

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


# SPILITING DATA INTO TRAINING AND TESTING

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_train.shape)
print(data_test.shape)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


data_train_array= scaler.fit_transform(data_train)


# Loading Model
model= load_model('keras_modelOpening.h5')

# Testing part

past_100_days = data_train.tail(100)
final_df = past_100_days.append(data_test, ignore_index=True)
input_data = scaler.fit_transform(final_df)

# Spliting Data into x_test and y_test
x_test=[]
y_test=[]


for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test =np.array(x_test), np.array(y_test)
y_predicted= model.predict(x_test)

scaler = scaler.scale_
scale_factor =1/scaler[0]
y_predicted =y_predicted* scale_factor
y_test = y_test * scale_factor


# Final Graph
st. markdown("<h1 style='text-align: center; color: White;'>Predictions VS Original</h1>", unsafe_allow_html=True)

fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Closing Price')
plt.plot(y_predicted, 'r', label= 'Predicted Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)