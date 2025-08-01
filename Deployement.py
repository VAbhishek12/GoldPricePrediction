import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("📈 Gold Price Forecasting using LSTM")

# Load Data
@st.cache_data
def load_data():
    df = web.DataReader('GLD', data_source='yahoo', start='2005-01-01', end='2021-06-01')
    df.index = pd.to_datetime(df.index)
    df_monthly = df['Close'].resample('M').last()
    return pd.DataFrame(df_monthly)

df = load_data()
st.subheader("Historical Monthly Gold Prices")
st.line_chart(df)

# Preprocessing
data = df.values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

def create_dataset(dataset, time_step=60):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=0)

# Prediction
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(real_prices, color='blue', label='Actual Price')
ax.plot(predicted_prices, color='red', label='Predicted Price')
ax.set_title("Gold Price Prediction")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
