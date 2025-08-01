import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import boxcox
import datetime
import warnings

warnings.filterwarnings("ignore")

st.title("📊 Gold Price Forecast App")


gold_ticker = "GC=F"
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.datetime.now() - datetime.timedelta(days=8*365)).strftime('%Y-%m-%d')

data = yf.download(gold_ticker, start=start_date, end=end_date)

if data.empty or 'Close' not in data.columns:
    st.error("Failed to fetch gold futures data. Try again later.")
    st.stop()

data = data.reset_index()
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data.rename(columns={'Close': 'Price'}, inplace=True)
data['Price'] = np.round(data['Price'], 2)

data1 = data.set_index('Date')
non_na_index = data1.index.dropna()

if non_na_index.empty:
    st.error("Invalid gold data. No valid dates found.")
    st.stop()

start_date = non_na_index.min()
end_date = non_na_index.max()
date_range = pd.date_range(start=start_date, end=end_date)

data1 = data1.reindex(date_range)
data1['Price'] = data1['Price'].interpolate(method='time')

data1_bymonthly = data1.resample('M').mean()

# ✅ Defensive checks before applying Box-Cox
if 'Price' not in data1_bymonthly.columns:
    st.error("Expected 'Price' column is missing in monthly data.")
    st.stop()

price_series = data1_bymonthly['Price']

if not isinstance(price_series, pd.Series) or price_series.ndim != 1:
    st.error("Monthly price data is not properly formatted.")
    st.stop()

if price_series.isnull().any():
    st.error("Monthly gold data contains missing values. Cannot proceed.")
    st.stop()

# Apply Box-Cox transformation
data_boxcox = pd.Series(boxcox(price_series, lmbda=0), index=price_series.index)


try:
    loaded_model = joblib.load("gold_forecast.joblib")
except Exception as e:
    st.error("Model loading failed. Ensure 'gold_forecast.joblib' is in the correct path.")
    st.stop()


year = int(st.number_input("Enter the year: ", placeholder="YYYY", value=2025, step=1))
month = int(st.number_input("Enter the month: ", placeholder="MM", value=8, step=1))

def last_day_of_month(year, month):
    return pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)

last_day = last_day_of_month(year, month)

# Forecast 1-step ahead
forecast_diff = np.round(loaded_model.forecast(steps=1), 6)
forecast_boxcox = forecast_diff.cumsum()
last_original_value = data_boxcox.iloc[-1]
forecast_boxcox = forecast_boxcox + last_original_value
forecast_price = np.round(np.exp(forecast_boxcox), 2)


if st.button("Submit"):
    st.success(f"📅 Forecasted Gold Price for {last_day.strftime('%B %Y')}: **{float(forecast_price.values)} USD**")
