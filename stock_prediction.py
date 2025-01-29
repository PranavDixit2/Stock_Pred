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
import logging
import streamlit as st
from keras_tuner import HyperModel, RandomSearch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dropout, Dense, BatchNormalization

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to fetch stock data with input validation
@st.cache
def fetch_stock_data(ticker):
    try:
        stock_data = yf.download(ticker)
        if stock_data.empty:
            raise ValueError("No data found. Please check the ticker symbol.")
        logging.info("Stock data fetched successfully.")
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        st.error(f"Error fetching stock data: {e}")
        return None

# Function to calculate EMA
def calculate_ema(data, span):
    return data['Close'].ewm(span=span, adjust=False).mean()

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
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

# Function to calculate additional features
def calculate_features(data):
    data['EMA_9'] = calculate_ema(data, 9)
    data['EMA_21'] = calculate_ema(data, 21)
    data['EMA_50'] = calculate_ema(data, 50)
    data['EMA_200'] = calculate_ema(data, 200)
    data['RSI'] = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_macd(data)

    # Volume change percentage
    data['Volume_Change'] = data['Volume'].pct_change()

    # Lagged values of the target variable and other features
    for lag in range(1, 6):  # Adding lagged values for the last 5 days
        data[f'Lag_Close_{lag}'] = data['Close'].shift(lag)
        data[f'Lag_Volume_{lag}'] = data['Volume'].shift(lag)
        data[f'Lag_EMA_9_{lag}'] = data['EMA_9'].shift(lag)
        data[f'Lag_RSI_{lag}'] = data['RSI'].shift(lag)
        data[f'Lag_MACD_{lag}'] = data['MACD'].shift(lag)
        data[f'Lag_Bollinger_Upper_{lag}'] = data['Bollinger_Upper'].shift(lag)
        data[f'Lag_Bollinger_Lower_{lag}'] = data['Bollinger_Lower'].shift(lag)

    data['Price_Change'] = data['Close'].pct_change()  # Price change percentage

    # Drop rows with NaN values
    data.dropna(inplace=True)
    logging.info("Features calculated.")
    return data

# Function to preprocess data
def preprocess_data(data):
    imputer = KNNImputer(n_neighbors=5)
    data[['Open', 'High', 'Low', 'Close', 'Volume']] = imputer.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    data['Target'] = data['Close'].pct_change().shift(-1)  # Predict the next day's percentage change
    data.dropna(inplace=True)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
                'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'MACD', 'MACD_Signal', 'Volume_Change'] + \
                [f'Lag_Close_{lag}' for lag in range(1, 6)] + \
                [f'Lag_Volume_{lag}' for lag in range(1, 6)] + \
                [f'Lag_EMA_9_{lag}' for lag in range(1, 6)] + \
                [f'Lag_RSI_{lag}' for lag in range(1, 6)] + \
                [f'Lag_MACD_{lag}' for lag in range(1, 6)] + \
                [f'Lag_Bollinger_Upper_{lag}' for lag in range(1, 6)] + \
                [f'Lag_Bollinger_Lower_{lag}' for lag in range(1, 6)] + \
                ['Price_Change']
    logging.info("Data after preprocessing.")
    return data, features

# Function to build and train LSTM model
class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        # Use a fixed number of features here or pass it as an argument
        num_features = 15  # Adjust this based on your feature set
        model.add(Bidirectional(GRU(hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=True), input_shape=(None, num_features)))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        model.add(Bidirectional(GRU(hp.Int('units', min_value=32, max_value=256, step=32)))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        model.add(Dense(1))  # Change to a single neuron for regression
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                      loss='mean_squared_error', metrics=['mae'])  # Use MSE for regression
        return model

# Function to train the model with hyperparameter tuning
def train_model_with_tuning(X_train, y_train, X_test, y_test):
    tuner = RandomSearch(LSTMHyperModel(), objective='val_mae', max_trials=5, executions_per_trial=3)
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

# Function to prepare data for LSTM
def prepare_data(data, features, window_size=10):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(data[features])
    logging.info("Scaled features prepared.")

    X = np.array([scaled_features[i-window_size:i] for i in range(window_size, len(scaled_features))])
    y = data['Target'].values[window_size:]

    # Normalize the target variable
    y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    logging.info(f"Prepared data for LSTM: X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_scaler, target_scaler

# Function to predict the next day's stock price
def predict_next_day(model, data, features, feature_scaler, target_scaler, window_size=10):
    last_data = data[features].values[-window_size:]
    last_data_scaled = feature_scaler.transform(last_data)
    last_data_reshaped = last_data_scaled.reshape((1, window_size, len(features)))

    predicted_change = model.predict(last_data_reshaped)[0][0]  # Directly get the predicted change
    predicted_price = data['Close'].values[-1] * (1 + predicted_change)  # Calculate the predicted price
    return predicted_price

# Function to calculate prediction intervals using quantile regression
def calculate_prediction_intervals(model, X_test, y_test, alpha=0.05):
    # Make predictions
    y_pred = model.predict(X_test).flatten()

    # Calculate residuals
    residuals = y_test - y_pred

    # Estimate the quantiles for the desired confidence level
    lower_bound = np.percentile(residuals, 100 * alpha / 2)
    upper_bound = np.percentile(residuals, 100 * (1 - alpha / 2))

    # Calculate lower and upper bounds of the prediction intervals
    lower_bound = y_pred + lower_bound
    upper_bound = y_pred + upper_bound

    return y_pred, lower_bound, upper_bound

# Streamlit application
def main():
    st.title("Stock Price Prediction with LSTM")

    ticker = st.text_input("Enter the stock ticker (e.g., AAPL):", "AAPL")

    if st.button("Fetch Data"):
        data = fetch_stock_data(ticker)

        if data is not None:
            # Filter to keep only the last 90 trading days
            data = data.tail(90)

            data = calculate_features(data)
            data, features = preprocess_data(data)

            # Prepare data for LSTM
            window_size = 10  # You can change this value to experiment
            X, y, feature_scaler, target_scaler = prepare_data(data, features)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model with hyperparameter tuning
            model = train_model_with_tuning(X_train, y_train, X_test, y_test)

            # Calculate prediction intervals
            y_pred, lower_bound, upper_bound = calculate_prediction_intervals(model, X_test, y_test)

            # Inverse transform the predictions and bounds
            y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            lower_bound = target_scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
            upper_bound = target_scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
            y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # Calculate performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100  # Avoid division by zero
            r_squared = r2_score(y_test, y_pred)

            st.write(f'MAE: {mae:.2f}')
            st.write(f'MSE: {mse:.2f}')
            st.write(f'RMSE: {rmse:.2f}')
            st.write(f'MAPE: {mape:.2f}%')
            st.write(f'R-squared: {r_squared:.2f}')

            # Plotting Actual vs Predicted with Prediction Intervals using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='Actual Prices', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred, mode='lines', name='Predicted Prices', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=lower_bound, mode='lines', name='Lower Bound', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=upper_bound, mode='lines', name='Upper Bound', line=dict(color='gray', dash='dash')))
            fig.update_layout(title=f'Actual vs Predicted Stock Prices for {ticker} with Prediction Intervals',
                              xaxis_title='Date',
                              yaxis_title='Price')
            st.plotly_chart(fig)

            # Predict the next day's stock price
            predicted_price = predict_next_day(model, data, features, feature_scaler, target_scaler, window_size)
            st.write(f"The predicted closing price for the next trading day for {ticker} is: {predicted_price:.2f}")

if __name__ == "__main__":
    main()
