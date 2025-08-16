
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from io import StringIO

st.set_page_config(layout="wide")
st.title("📈 Rolling Forecast Web App for OTA KPIs")

# --- Sidebar ---
st.sidebar.header("Upload Historical Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

product = st.sidebar.selectbox("Select Product", ["Flights", "DP", "Hotel", "TO"])
metric = st.sidebar.selectbox("Select KPI", ["Bookings", "GMV", "Revenue", "Marketing Cost", "Margin", "Gross Profit", "Variable Cost", "Voucher Discount"])

# --- Helper Functions ---
def load_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def arima_forecast(df, periods):
    model = ARIMA(df['value'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    forecast_index = pd.date_range(start=df['date'].iloc[-1] + pd.offsets.MonthBegin(), periods=periods, freq='MS')
    return pd.DataFrame({'date': forecast_index, 'forecast': forecast})

def prophet_forecast(df, periods):
    df_prophet = df.rename(columns={'date': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'forecast'})[-periods:]

# --- Main ---
if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader(f"📊 Input Time Series for {product} - {metric}")
    st.line_chart(df.set_index('date'))

    forecast_period = st.slider("Forecast Months (YTG)", min_value=1, max_value=12, value=7)
    model_type = st.selectbox("Choose Forecasting Model", ["ARIMA", "Prophet"])

    st.subheader("🎯 Scenario Planning")
    scenario_adj = st.slider("Adjust KPI by %", min_value=-50, max_value=50, value=0, step=5)

    if st.button("Run Forecast"):
        with st.spinner("Forecasting..."):
            df['value'] *= (1 + scenario_adj / 100)

            if model_type == "ARIMA":
                forecast_df = arima_forecast(df, forecast_period)
            else:
                forecast_df = prophet_forecast(df, forecast_period)

            combined = pd.concat([
                df.rename(columns={"value": "forecast"})[['date', 'forecast']],
                forecast_df
            ]).reset_index(drop=True)

            st.subheader("📉 Forecast Output")
            fig, ax = plt.subplots()
            ax.plot(combined['date'], combined['forecast'], label='Forecast')
            ax.axvline(x=df['date'].iloc[-1], color='red', linestyle='--', label='Forecast Start')
            ax.set_title(f"Rolling Forecast: {product} - {metric}")
            ax.legend()
            st.pyplot(fig)

            st.dataframe(forecast_df.set_index('date'))

            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast CSV", csv, file_name="forecast_output.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file with 'date' and 'value' columns.")
