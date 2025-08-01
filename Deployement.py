import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Title
st.title('Gold Price Prediction using LSTM (Live Deployment)')

# Load data from Yahoo Finance
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()

df = yf.download('GC=F', start=start, end=end)  # Gold Futures symbol
df = df[['Close']]  # Use only closing prices
df = df.dropna()

st.subheader('Historical Gold Prices (Close)')
st.line_chart(df['Close'])

# Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Prepare testing data
prediction_days = 60
X_test = []
y_test = []

for i in range(prediction_days, len(scaled_data)):
    X_test.append(scaled_data[i - prediction_days:i, 0])
    y_test.append(scaled_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Load pre-trained model
model = load_model('gold_price_model.h5')

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotting
st.subheader('Predicted vs Actual Prices')
fig, ax = plt.subplots()
ax.plot(real_prices, color='blue', label='Actual Price')
ax.plot(predictions, color='red', label='Predicted Price')
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Latest Prediction
latest_data = scaled_data[-prediction_days:]
latest_data = np.array(latest_data)
latest_data = np.reshape(latest_data, (1, latest_data.shape[0], 1))

predicted_price = model.predict(latest_data)
predicted_price = scaler.inverse_transform(predicted_price)

st.success(f"📈 Predicted Gold Price for Tomorrow: **${predicted_price[0][0]:.2f}**")
