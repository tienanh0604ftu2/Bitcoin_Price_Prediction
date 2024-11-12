import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os

sns.set_theme()

# Setup the interface with light/dark mode
st.set_page_config(page_title="Bitcoin Price Prediction App", page_icon="\U0001F4B8", layout="wide")
theme_option = st.sidebar.selectbox("Choose display mode:", ["Light", "Dark"])

# Apply light/dark mode styling
if theme_option == "Dark":
    plt.style.use('dark_background')
    st.markdown(
        """
        <style>
            body {
                background-color: #2E2E2E;
                color: white;
            }
            .css-1v0mbdj, .css-18ni7ap {
                background-color: #2E2E2E;
                color: white;
            }
            .st-df {
                background-color: #2E2E2E;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    plt.style.use('default')
    st.markdown(
        """
        <style>
            body {
                background-color: white;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("💰 Bitcoin Price Prediction Application")
st.markdown("### 🚀 **Experience future Bitcoin price prediction with a modern and user-friendly interface**")
st.markdown("**Please enter the necessary parameters and view the visual results!**")

# Sidebar
st.sidebar.header("⚙️ Application Settings")
st.sidebar.markdown("Adjust parameters to modify prediction results")

# Load Bitcoin data
from datetime import datetime
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)
stock = "BTC-USD"
bit_coin_data = yf.download(stock, start, end)

st.sidebar.subheader("🗓️ Data Time Range")
st.sidebar.write(f"Start: {start.strftime('%d-%m-%Y')}")
st.sidebar.write(f"End: {end.strftime('%d-%m-%Y')}")

st.subheader("📈 Bitcoin Data")
st.dataframe(bit_coin_data.style.format("{:.2f}").highlight_max(axis=0))

# Load model
model_path = 'models/neural_network.h5'
if os.path.exists(model_path):
    try:
        # Load the model using TensorFlow's load_model method
        model = tf.keras.models.load_model(model_path, compile=False)
    except (OSError, ImportError, ValueError):
        st.error("Error: Unable to load model. The model file may be corrupted or invalid, or it might not be compatible with the current Keras version.")
        st.stop()
else:
    st.error("Error: Model file not found. Please check that 'neural_network.h5' exists in the 'models' directory.")
    st.stop()

# Split data
splitting_len = int(len(bit_coin_data) * 0.9)
x_test = pd.DataFrame(bit_coin_data["Close"][splitting_len:])

# Original close price plot
st.subheader("📊 Original Close Price")
fig = plt.figure(figsize=(15, 6))
plt.plot(bit_coin_data["Close"], color='skyblue', linestyle='-', linewidth=2)
plt.title("Bitcoin Close Price Over the Years", fontsize=20)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Close Price (USD)", fontsize=14)
st.pyplot(fig)

# Test close price data
st.subheader("🔍 Test Close Price")
st.dataframe(x_test.style.format("{:.2f}").highlight_max(axis=0))

fig = plt.figure(figsize=(15, 6))
plt.plot(x_test, color='green', linestyle='-', linewidth=2)
plt.title("Test Close Price", fontsize=20)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Close Price (USD)", fontsize=14)
st.pyplot(fig)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[["Close"].values])

# Prepare sequence for model prediction
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Predict data
predictions = model.predict(x_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Data for plotting
plotting_data = pd.DataFrame({
    "Original Test Data": inv_y_test.reshape(-1),
    "Predictions": inv_predictions.reshape(-1)
})
index = bit_coin_data.index[splitting_len + 100:]

st.subheader("📉 Original and Predicted Price")
st.write(plotting_data)

fig = plt.figure(figsize=(15, 6))
plt.plot(plotting_data["Original Test Data"], color='orange', linestyle='-', linewidth=2, label="Original Test Data")
plt.plot(plotting_data["Predictions"], color='blue', linestyle='--', linewidth=2, label="Predicted Data")
plt.title("Original and Predicted Price", fontsize=20)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Close Price (USD)", fontsize=14)
plt.legend()
st.pyplot(fig)

# Future close price prediction
st.subheader("🔮 Future Close Price")
last_100 = bit_coin_data[["Close"]].tail(100)
last_100 = scaler.fit_transform(last_100["Close"].values.reshape(-1, 1)).reshape(1, -1, 1)

def predict_future(no_of_days, prev_100):
    future_predictions = []
    for _ in range(no_of_days):
        next_day = model.predict(prev_100)  # No conversion to list here
        prev_100 = np.concatenate([prev_100[:, 1:, :], next_day.reshape(1, 1, -1)], axis=1)
        future_predictions.append(scaler.inverse_transform(next_day))
    return future_predictions

no_of_days = st.sidebar.slider("Select number of days to predict", min_value=1, max_value=30, value=10, step=1)
future_result = predict_future(no_of_days, last_100)
future_result = np.array(future_result).reshape(-1, 1)

fig = plt.figure(figsize=(15, 6))
plt.plot(pd.DataFrame(future_result), marker="o", color='purple')
for i in range(len(future_result)):
    plt.text(i, future_result[i], int(future_result[i][0]), fontsize=12, color='black')
plt.xlabel("Future Days", fontsize=14)
plt.ylabel("Close Price (USD)", fontsize=14)
plt.title(f"Future Close Price of Bitcoin for {no_of_days} Days", fontsize=20)
st.pyplot(fig)

# User feedback
st.success("✅ Prediction Complete! Thank you for using our application.")
st.balloons()

# Add contact link
st.sidebar.markdown("### 📧 **Contact Us**")
st.sidebar.markdown("If you have any questions, feel free to contact us via email: [support@example.com](mailto:support@example.com)")
