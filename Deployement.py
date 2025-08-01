import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Gold Price Forecast App", page_icon=":bar_chart:")

st.title("📊 Gold Price Forecast App")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Upload monthly gold price data
uploaded_file = st.file_uploader("Upload Monthly Gold Price CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Try to infer the date column
        date_col = None
        for col in df.columns:
            if pd.to_datetime(df[col], errors='coerce').notna().sum() > 0:
                date_col = col
                break

        if date_col is None:
            st.error("No date column found in the uploaded file.")
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df.sort_values(by=date_col, inplace=True)

            # Assume the price column is the second column
            price_col = [col for col in df.columns if col != date_col][0]
            price_series = df[price_col]

            if not isinstance(price_series, pd.Series):
                st.error("Price column is not a valid series.")
            elif price_series.isnull().any():
                st.error("Price column contains missing values.")
            else:
                st.success("Data successfully loaded and validated.")

                # Plot actual prices
                st.subheader("📈 Actual Gold Prices")
                plt.figure(figsize=(10, 4))
                plt.plot(df[date_col], price_series)
                plt.xlabel("Date")
                plt.ylabel("Gold Price")
                st.pyplot(plt)

                # Forecasting
                st.subheader("🔮 Forecast Next Month's Gold Price")
                last_price = price_series.iloc[-1]
                prediction = model.predict([[last_price]])
                st.metric(label="Predicted Price", value=f"{prediction[0]:,.2f} INR")

    except Exception as e:
        st.error(f"Error reading or processing file: {e}")
else:
    st.info("Please upload a CSV file containing monthly gold prices.")
