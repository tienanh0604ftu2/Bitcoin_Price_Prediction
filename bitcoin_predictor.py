import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
sns.set_theme()

st.title("Bitcoin Price Predictor Application")

from datetime import datetime
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)
stock = "BTC-USD"
bit_coin_data = yf.download(stock, start, end)

st.subheader("Bit coin Data")
st.write(bit_coin_data)

model = load_model("Latest_Bitcoin_Model.keras")

splitting_len = int(len(bit_coin_data) * 0.9)

x_test = pd.DataFrame(bit_coin_data["Close"][splitting_len:])

st.subheader("Original Close Price")
figsize = (15,6)
fig = plt.figure(figsize = figsize)
plt.plot(bit_coin_data["Close"], "b")
st.pyplot(fig)

st.subhead("Test Close Price")
st.write(x_test)

figsize = (15,6)
fig = plt.figure(figsize = figsize)
plt.plot(x_test, "b")
st.pyplot(fig)

scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(x_test[["Close"]].values)

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_test)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame({
    "Original_test_data": inv_y_test.reshape(-1),
    "Predictions": inv_predictions.reshape(-1)
})
index = bit_coin_data.index[splitting_len + 100:]

st.subheader("Original Price and Predicted Price")
st.write(plotting_data)

st.subheader("Original Price and Predicted Price")
fig = plt.figure(figsize = figsize)
plt.plot(pd.concat([bit_coin_data.Close[:splitting_len + 100], plotting_data], axis = 0))
plt.legend("Data Not Used", "Original Test Data", "Predicted Test Data")
st.pyplot(fig)

st.subheader("Future Close Price Value")
last_100 = bit_coin_data[["Close"]].tail(100)
last_100 = scaler.fit_transform(last_100["Close"].values.reshape(-1, 1)).reshape(1, -1, 1)

def predict_future(no_of_days, prev_100):
    future_predictions = []
    for _ in range(no_of_days):
        # Dự đoán ngày tiếp theo từ mô hình
        next_day = model.predict(prev_100)  # Không chuyển đổi thành danh sách

        # Cập nhật prev_100: Loại bỏ phần tử đầu tiên và thêm next_day vào cuối
        prev_100 = np.concatenate([prev_100[:, 1:, :], next_day.reshape(1, 1, -1)], axis=1)
        
        # Thêm dự đoán vào danh sách future_predictions
        future_predictions.append(scaler.inverse_transform(next_day))
    
    return future_predictions

no_of_days = st.input_text("Enter the No of days to be predicted from current data: ", "10")
future_result = predict_future(no_of_days, last_100)
future_result = np.array(future_result).reshape(-1,1)

fig = plt.figure(figsize = (15,6))
plt.plot(pd.DataFrame(future_result), marker = "o")
for i in range(len(future_result)):
    plt.text(i, future_result[i], int(future_result[i][0]))
plt.xlabel("Future Days")
plt.ylabel("Close Price")
plt.title("Feture Close Price of Bit Coin")
st.pyplot(fig)

