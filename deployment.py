import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import boxcox
import datetime 
import warnings

warnings.filterwarnings("ignore")


# Define the ticker symbol for gold and download gold data
gold_ticker = "GC=F"  
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.datetime.now() - datetime.timedelta(days=8*365)).strftime('%Y-%m-%d')
data = yf.download(gold_ticker, start=start_date, end=end_date)
data = data.reset_index()
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data.rename(columns = {'Close': 'Price'}, inplace = True)
data['Price'] = np.round(data['Price'], 2)
data1 = data.set_index('Date')
start_date = data1.index.min()
end_date = data1.index.max()
date_range = pd.date_range(start=start_date, end=end_date)
data1 = data1.reindex(date_range)
data1_bymonthly = data1.resample('M').mean()
data_boxcox = pd.Series(boxcox(data1_bymonthly['Price'], lmbda=0), index = data1_bymonthly.index)


# Load the pre-trained ARIMA model
loaded_model = joblib.load('gold_forecast.joblib')

# Function to find the last day of a given month and year
def last_day_of_month(year, month):
    return pd.Timestamp(year,month, 1) + pd.offsets.MonthEnd(0)

st.title("Gold Price Predition")
# Take input for year and month
year = int(st.number_input("Enter the year: ", placeholder = "Type in YYYY format", value = 2024, step = 1))
month = int(st.number_input("Enter the month: ", placeholder = "Type in MM format...", value = 5, step = 1))


# Find the last day of the given month and year
last_day = last_day_of_month(year, month)
last_day_formatted = last_day.strftime('%Y-%m-%d')

# Forecast gold price for the last day of the given month and year
forecast_steps = (last_day - pd.Timestamp.now()).days
forecast_diff = np.round(loaded_model.forecast(steps=forecast_steps), 6)

# Inverse differencing
forecast_boxcox = forecast_diff.cumsum()
last_original_value = data_boxcox.iloc[-1]
forecast_boxcox = forecast_boxcox.add(last_original_value)

# Inverse Box-Cox transformation
forecast = np.round(np.exp(forecast_boxcox), 2)


button = st.button("Submit")

if button:
    st.text(f"Forecasted Monthly Gold Futures Price: {forecast[last_day_formatted]} USD") 
