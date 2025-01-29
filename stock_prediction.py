import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import KNNImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, BatchNormalization
import logging
import streamlit as st
from kerastuner.tuners import RandomSearch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to fetch stock data
def fetch_stock_data(ticker):
    try:
        stock_data = yf.download(ticker)
        if stock_data.empty:
            raise ValueError("No data found. Please check the ticker symbol.")
        logging.info(f"Stock data for {ticker} fetched successfully.")
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        return None

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)  # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    data['Bollinger_Upper'] = sma + (2 * std_dev)
    data['Bollinger_Lower'] = sma - (2 * std_dev)
    return data

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# Function to calculate features
def calculate_features(data):
    for span in [9, 21, 50, 200]:
        data[f'EMA_{span}'] = data['Close'].ewm(span=span, adjust=False).mean()

    data['RSI'] = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_macd(data)
    data['Volume_Change'] = data['Volume'].pct_change()

    for lag in range(1, 6):
        data[f'Lag_Close_{lag}'] = data['Close'].shift(lag)

    data['Price_Change'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    logging.info("Features calculated successfully.")
    return data

# Function to preprocess data
def preprocess_data(data):
    imputer = KNNImputer(n_neighbors=5)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data.loc[:, numeric_cols] = imputer.fit_transform(data[numeric_cols])

    data['Target'] = data['Close'].pct_change().shift(-1)
    data.dropna(inplace=True)

    features = [col for col in data.columns if col not in ['Target']]
    logging.info("Data preprocessed successfully.")
    return data, features

# HyperModel Class for LSTM
class LSTMHyperModel:
    def __init__(self, feature_count):
        self.feature_count = feature_count

    def build(self, hp):
        model = Sequential([
            Bidirectional(GRU(hp.Int('units', 32, 256, 32), return_sequences=True), 
                          input_shape=(None, self.feature_count)),
            BatchNormalization(),
            Dropout(hp.Float('dropout', 0.1, 0.5, 0.1)),
            Bidirectional(GRU(hp.Int('units', 32, 256, 32))),
            BatchNormalization(),
            Dropout(hp.Float('dropout', 0.1, 0.5, 0.1)),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, 'LOG')),
                      loss='mean_squared_error', metrics=['mae'])
        return model

# Function to train model with tuning
def train_model_with_tuning(X_train, y_train, X_test, y_test, feature_count):
    tuner = RandomSearch(LSTMHyperModel(feature_count).build, objective='val_mae', max_trials=5, executions_per_trial=3)
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    best_model = tuner.get_best_models(1)[0]
    return best_model

# Function to prepare LSTM data
def prepare_data(data, features, window_size=10):
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[features])
    
    X = np.array([scaled_features[i-window_size:i] for i in range(window_size, len(scaled_features))])
    y = data['Target'].values[window_size:]

    logging.info(f"Prepared LSTM data: X shape {X.shape}, y shape {y.shape}")
    return X, y, scaler

# Function to predict next day stock price
def predict_next_day(model, data, features, scaler, window_size=10):
    if len(data) < window_size:
        logging.error("Insufficient data for prediction.")
        return None

    last_data = scaler.transform(data[features].values[-window_size:])
    last_data_reshaped = last_data.reshape((1, window_size, len(features)))

    predicted_change = model.predict(last_data_reshaped)[0][0]
    predicted_price = data['Close'].values[-1] * (1 + predicted_change)
    return predicted_price

# Streamlit app
def main():
    st.title("Stock Price Prediction with LSTM")
    ticker = st.text_input("Enter stock ticker:", "AAPL")

    if st.button("Fetch Data"):
        data = fetch_stock_data(ticker)
        if data is not None:
            data = calculate_features(data)
            data, features = preprocess_data(data)
            st.write(data.tail())

if __name__ == "__main__":
    main()
