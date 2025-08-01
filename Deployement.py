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
start_date = (datetime.datetime.now() - datetime.timedelta(days=8 * 365)).strftime('%Y-%m-%d')

data = yf.download(gold_ticker, start=start_date, end=end_date)

if data.empty or 'Close' not in data.columns:
    st.error("Failed to fetch gold futures data. Try again later.")
    st.stop()

data = data.reset_index()
data = data[['Date', 'Close']]
data.rename(columns={'Close': 'Price'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data['Price'] = np.round(data['Price'], 2)

# Set index and interpolate
data = data.set_index('Date')
data['Price'] = data['Price'].interpolate(method='time')

# Resample monthly average – clean Series
price_series = data['Price'].resample('M').mean()

if price_series.isnull().any():
    st.error("Monthly gold price contains missing values.")
    st.stop()

# Apply Box-Cox
data_boxcox = pd.Series(boxcox(price_series, lmbda=0), index=price_series.index)


try:
    loaded_model = joblib.load("gold_forecast.joblib")
except Exception as e:
    st.error("Model loading failed. Ensure 'gold_forecast.joblib' is present.")
    st.stop()


year = int(st.number_input("Enter the year: ", placeholder="YYYY", value=2025, step=1))
month = int(st.number_input("Enter the month: ", placeholder="MM", value=8, step=1))

def last_day_of_month(year, month):
    return pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)

last_day = last_day_of_month(year, month)

# Forecast
forecast_diff = np.round(loaded_model.forecast(steps=1), 6)
forecast_boxcox = forecast_diff.cumsum()
last_original_value = data_boxcox.iloc[-1]
forecast_boxcox = forecast_boxcox + last_original_value
forecast_price = np.round(np.exp(forecast_boxcox), 2)


if st.button("Submit"):
    st.success(f"📅 Forecasted Gold Price for {last_day.strftime('%B %Y')}: **{float(forecast_price.values)} USD**")
