# Deployement.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression

# Title
st.set_page_config(page_title="Gold Price Forecast App", page_icon="📊")
st.title("📊 Gold Price Forecast App")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("monthly_gold_prices.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date', 'Price'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("gold_price_model.pkl")
    return model

# Main
df = load_data()

# Format and display the data
st.subheader("Monthly Gold Price Data")
st.dataframe(df)

# Ensure proper formatting
try:
    price_series = df['Price']
    if price_series.isnull().any():
        st.error("Price column contains null values.")
        st.stop()
except Exception as e:
    st.error(f"Monthly price data is not properly formatted. Error: {str(e)}")
    st.stop()

# Load the trained model
model = load_model()

# Forecasting future prices
st.subheader("🔮 Forecast Future Price")
months = st.slider("Select number of future months to forecast", 1, 24, 12)

# Prepare input for prediction
df['MonthIndex'] = range(1, len(df) + 1)
last_index = df['MonthIndex'].iloc[-1]
future_months = pd.DataFrame({'MonthIndex': range(last_index + 1, last_index + months + 1)})

# Predict
predictions = model.predict(future_months)

# Show predictions
st.subheader("📈 Forecasted Prices")
future_df = pd.DataFrame({
    "MonthIndex": future_months['MonthIndex'],
    "Forecasted Price": predictions
})
st.dataframe(future_df)

# Plot
fig, ax = plt.subplots()
ax.plot(df['MonthIndex'], df['Price'], label="Historical Price")
ax.plot(future_df['MonthIndex'], future_df['Forecasted Price'], label="Forecasted Price", linestyle="--")
ax.set_xlabel("Month Index")
ax.set_ylabel("Gold Price")
ax.set_title("Gold Price Forecast")
ax.legend()
st.pyplot(fig)
