import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("final_model.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌍 Air Quality Index (AQI) Prediction")
st.write(
    "Enter the pollutant concentrations below to predict the Air Quality Index."
)


# --------------------------------------------------
# Input parameters
# --------------------------------------------------

st.subheader("Enter Pollution Parameters")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5", min_value=0.0, value=80.0)
    pm10 = st.number_input("PM10", min_value=0.0, value=120.0)
    no = st.number_input("NO", min_value=0.0, value=20.0)
    no2 = st.number_input("NO2", min_value=0.0, value=35.0)
    nox = st.number_input("NOx", min_value=0.0, value=45.0)
    nh3 = st.number_input("NH3", min_value=0.0, value=10.0)

with col2:
    co = st.number_input("CO", min_value=0.0, value=0.8)
    so2 = st.number_input("SO2", min_value=0.0, value=15.0)
    o3 = st.number_input("O3", min_value=0.0, value=30.0)
    benzene = st.number_input("Benzene", min_value=0.0, value=2.0)
    toluene = st.number_input("Toluene", min_value=0.0, value=5.0)
    xylene = st.number_input("Xylene", min_value=0.0, value=1.0)


# --------------------------------------------------
# Create input dataframe
# --------------------------------------------------

input_data = pd.DataFrame({
    "PM2.5": [pm25],
    "PM10": [pm10],
    "NO": [no],
    "NO2": [no2],
    "NOx": [nox],
    "NH3": [nh3],
    "CO": [co],
    "SO2": [so2],
    "O3": [o3],
    "Benzene": [benzene],
    "Toluene": [toluene],
    "Xylene": [xylene]
})


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict AQI"):

    predicted_aqi = model.predict(input_data)[0]

    st.success("Prediction completed successfully!")

    st.metric(
        label="Predicted AQI",
        value=f"{predicted_aqi:.2f}"
    )