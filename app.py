import streamlit as st
import yfinance as yf
import numpy as np
import pickle
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

model = load_model('stock_prediction_model.keras')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Stock Price Prediction using LSTM")
st.markdown("Enter a stock symbol (e.g., AAPL) to predict its future stock prices.")

stock_symbol = st.text_input("Stock Symbol", "AAPL")

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
df = yf.download(stock_symbol, start=start, end=end)

st.subheader(f"Stock data for {stock_symbol}")
st.write(df.tail())


data = df[['Close']]
scaled_data = scaler.transform(data.values)


training_data_len = int(np.ceil(len(scaled_data) * 0.95))
test_data = scaled_data[training_data_len - 60:]
x_test = []

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 6))
plt.title(f'{stock_symbol} Stock Price Prediction', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train.index, train['Close'], label='Training Data')
plt.plot(valid.index, valid['Close'], label='Actual Prices')
plt.plot(valid.index, valid['Predictions'], label='Predictions')
plt.legend(loc='lower right')
st.pyplot(plt)

st.subheader('Predicted Stock Prices')
st.write(valid[['Close', 'Predictions']].tail())


st.subheader("Next 10 Days Prediction")
last_60_days = scaled_data[-60:]
next_10_days = []

for _ in range(10):
    x_input = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    pred = model.predict(x_input)
    next_10_days.append(pred[0, 0])
    last_60_days = np.append(last_60_days[1:], pred, axis=0)

next_10_days = scaler.inverse_transform(np.array(next_10_days).reshape(-1, 1))

next_dates = [end + timedelta(days=i) for i in range(1, 11)]
next_10_days_df = pd.DataFrame({'Date': next_dates, 'Predicted Close': next_10_days.flatten()})
st.write(next_10_days_df)

plt.figure(figsize=(10, 5))
plt.title('Next 10 Days Stock Price Prediction')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price USD ($)', fontsize=12)
plt.plot(next_10_days_df['Date'], next_10_days_df['Predicted Close'], marker='o', label='Predicted Prices')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)