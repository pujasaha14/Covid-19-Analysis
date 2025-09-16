import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="COVID-19 Analysis", layout="wide")

st.title("COVID-19 Data Analysis Dashboard")

# Fetch data from OWID
@st.cache_data
def load_data():
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    try:
        df = pd.read_csv(url, parse_dates=["date"])
        return df
    except Exception as e:
        st.warning("⚠️ Live fetch failed. Using bundled local dataset instead.")
        try:
            return pd.read_csv("owid-covid-data.csv", parse_dates=["date"])
        except Exception as e2:
            st.error(f"Failed to load both live and local data: {e2}")
            return pd.DataFrame()


# Sidebar
country = st.sidebar.selectbox("Select Country", df["location"].unique())
metric = st.sidebar.selectbox("Select Metric", ["total_cases", "total_deaths", "total_vaccinations"])

country_df = df[df["location"] == country]

# Line chart
st.subheader(f"{metric.replace('_', ' ').title()} in {country}")
fig = px.line(country_df, x="date", y=metric, title=f"{country} {metric}")
st.plotly_chart(fig, use_container_width=True)

# Summary stats
st.subheader("Summary Statistics")
st.write(country_df[[metric]].describe())

# Option: Forecast with Prophet (optional)
try:
    from prophet import Prophet

    forecast_button = st.sidebar.button("Run Forecast")
    if forecast_button:
        data = country_df[["date", metric]].rename(columns={"date": "ds", metric: "y"}).dropna()
        model = Prophet()
        model.fit(data)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig_forecast = px.line(forecast, x="ds", y="yhat", title=f"30-Day Forecast for {metric}")
        st.plotly_chart(fig_forecast, use_container_width=True)
except ImportError:
    st.warning("Prophet not installed. Run `pip install prophet` to enable forecasting.")
