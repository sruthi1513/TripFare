import streamlit as st
import numpy as np
import joblib
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Urban Taxi Fare Estimator", page_icon="🚖")
st.title("🚖 Urban Taxi Fare Estimator")
st.markdown("Predict the total taxi fare by providing trip details.")

def haversine(lat1, lon1, lat2, lon2):
    lon1, lon2, lat1, lat2 = map(radians, [lon1, lon2, lat1, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

with st.form("fare_form"):
    passenger_count = st.slider("Passenger Count", 1, 6, 1)

    col1, col2 = st.columns(2)
    with col1:
        pickup_lat = st.number_input("Pickup Latitude", value=40.7614327)
        pickup_lon = st.number_input("Pickup Longitude", value=-73.9798156)
    with col2:
        dropoff_lat = st.number_input("Dropoff Latitude", value=40.6513111)
        dropoff_lon = st.number_input("Dropoff Longitude", value=-73.8803331)

    pickup_date = st.date_input("Pickup Date", datetime.today())
    pickup_time = st.time_input("Pickup Time", datetime.now().time())

    submit = st.form_submit_button("Predict Fare")

if submit:
    pickup_datetime = datetime.combine(pickup_date, pickup_time)

    distance_km = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    avg_speed_kmph = 20
    duration_min = distance_km / avg_speed_kmph * 60

    log_distance = np.log1p(distance_km)
    log_duration = np.log1p(duration_min)

    X_input = np.array([[log_distance, log_duration]])
    predicted_log_fare = model.predict(X_input)[0]
    predicted_total_fare = np.expm1(predicted_log_fare)

    st.success(f"💰 Estimated Total Fare: **${predicted_total_fare:.2f}**")

    with st.expander("🔍 Prediction Details"):
        st.write(f"Trip Distance: `{distance_km:.2f} km`")
        st.write(f"Estimated Duration: `{duration_min:.2f} minutes`")
        st.write(f"Pickup Time: `{pickup_datetime}`")
