import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.title("Gold Price Prediction App")

# Load model
model = load_model("model.h5")

# Get historical gold prices
@st.cache_data
def load_data():
    df = yf.download('GC=F', start='2010-01-01', end='2023-12-31')
    return df[['Close']]

df = load_data()

st.subheader("Historical Gold Price")
st.line_chart(df)

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Prepare test data (last 100 days)
past_days = 100
x_test = []
for i in range(past_days, len(scaled_data)):
    x_test.append(scaled_data[i - past_days:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict and inverse scale
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Show prediction
df['Predicted'] = np.nan
df.iloc[-len(predictions):, df.columns.get_loc("Predicted")] = predictions.flatten()

st.subheader("Actual vs Predicted")
fig, ax = plt.subplots()
df[['Close', 'Predicted']].plot(ax=ax)
st.pyplot(fig)
