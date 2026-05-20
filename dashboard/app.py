import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# Auto refresh every 5 seconds
st_autorefresh(interval=5000, key="datarefresh")

# Page config
st.set_page_config(
    page_title="Precision Agriculture Dashboard",
    layout="wide"
)

# Title
st.title("🌾 Precision Agriculture Analytics Dashboard")

# Load trained AI model
model = joblib.load("../models/irrigation_model.pkl")

# Load crop disease detection model
disease_model = tf.keras.models.load_model(
    "../crop_disease/crop_disease_model.h5"
)

# Read sensor data
FILE_PATH = "../sensor_simulation/sensor_data.csv"

df = pd.read_csv(FILE_PATH)

# Show latest values
latest_data = df.iloc[-1]

# Prepare latest sensor values for AI prediction
features = [[
    latest_data["soil_moisture"],
    latest_data["temperature"],
    latest_data["humidity"],
    latest_data["soil_pH"],
    latest_data["light_intensity"]
]]

# AI prediction
prediction = model.predict(features)[0]

# Sidebar
st.sidebar.header("Farm Status")

# AI Recommendation Section
st.sidebar.subheader("🤖 AI Recommendation")

if prediction == 1:
    st.sidebar.error("🚨 AI Prediction: Irrigation Needed")

else:
    st.sidebar.success("✅ AI Prediction: No Irrigation Needed")

# Smart Recommendations
if latest_data["soil_moisture"] < 30:
    st.sidebar.error("⚠ Irrigation Required")

elif latest_data["soil_moisture"] > 70:
    st.sidebar.warning("⚠ Overwatering Risk")

else:
    st.sidebar.success("✅ Soil Moisture Optimal")

if latest_data["temperature"] > 35:
    st.sidebar.warning("🔥 High Temperature Detected")

if latest_data["humidity"] < 45:
    st.sidebar.warning("💧 Low Humidity Warning")

# Metric cards
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Soil Moisture",
    f"{latest_data['soil_moisture']}%"
)

col2.metric(
    "Temperature",
    f"{latest_data['temperature']} °C"
)

col3.metric(
    "Humidity",
    f"{latest_data['humidity']}%"
)

col4.metric(
    "Soil pH",
    latest_data['soil_pH']
)

st.markdown("---")

# Charts Section
st.subheader("📊 Real-Time Farm Analytics")

# Temperature chart
temp_chart = px.line(
    df,
    x="timestamp",
    y="temperature",
    title="Temperature Trend"
)

st.plotly_chart(temp_chart, use_container_width=True)

# Moisture chart
moisture_chart = px.line(
    df,
    x="timestamp",
    y="soil_moisture",
    title="Soil Moisture Trend"
)

st.plotly_chart(moisture_chart, use_container_width=True)

# Humidity chart
humidity_chart = px.line(
    df,
    x="timestamp",
    y="humidity",
    title="Humidity Trend"
)

st.plotly_chart(humidity_chart, use_container_width=True)

# Raw data section
st.subheader("📁 Live Sensor Data")

st.dataframe(df.tail(10))     

st.markdown("---")

st.subheader("🌿 AI Crop Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a Leaf Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Display uploaded image
    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Leaf Image", width=250)

    # Preprocess image
    img = img.resize((128, 128))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Predict disease
    prediction = disease_model.predict(img_array)

    prediction_value = prediction[0][0]

    st.subheader("🧠 AI Prediction Result")

    if prediction_value > 0.5:

        st.error("🚨 Disease Detected: Early Blight")

        st.warning(
            "Recommendation: Apply fungicide treatment and monitor crop health."
        )

    else:

        st.success("✅ Crop Appears Healthy")

        st.info(
            "Recommendation: Maintain current irrigation and nutrient levels."
        )

    st.write(f"Prediction Confidence: {prediction_value:.2f}")