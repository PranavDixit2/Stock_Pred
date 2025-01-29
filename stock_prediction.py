import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import streamlit as st

# Define the model file path in the models directory
MODEL_DIR = 'models'
MODEL_FILE_PATH = os.path.join(MODEL_DIR, 'lstm_stock_model.h5')

# Function to fetch stock data with input validation
def fetch_stock_data(ticker):
    try:
        stock_data = yf.download(ticker)
        if stock_data.empty:
            raise ValueError("No data found. Please check the ticker symbol.")
        return stock_data
    except Exception as e:
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
    return data

# Function to preprocess data
def preprocess_data(data):
    data['Target'] = data['Close'].shift(-1)  # Predict the next day's closing price
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
    return data, features

# Function to build and train LSTM model
def train_lstm_model(X_train, y_train, X_test, y_test, learning_rate=0.001, batch_size=64, epochs=100):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1], X_train.shape[2]))),
        Dropout(0.5),
        Bidirectional(LSTM(64, kernel_regularizer=l2(0.01))),
        Dropout(0.5),
        Dense(1)  # Change to a single neuron for regression
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error', metrics=['mae'])  # Use MSE for regression
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
    )
    model.save(MODEL_FILE_PATH)  # Save the model after training
    return model

# Function to prepare data for LSTM
def prepare_data(data, features, window_size=10):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(data[features])

    X = np.array([scaled_features[i-window_size:i] for i in range(window_size, len(scaled_features))])
    y = data['Target'].values[window_size:]

    # Normalize the target variable
    y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    return X, y, feature_scaler, target_scaler

# Function to predict the next day's stock price
def predict_next_day(model, data, features, feature_scaler, target_scaler, window_size=10):
    last_data = data[features].values[-window_size:]
    last_data_scaled = feature_scaler.transform(last_data)
    last_data_reshaped = last_data_scaled.reshape((1, window_size, len(features)))

    predicted_price = model.predict(last_data_reshaped)[0][0]  # Directly get the predicted price
    predicted_price = target_scaler.inverse_transform([[predicted_price]])[0][0]  # Inverse transform to get actual price
    return predicted_price

# Function to calculate prediction intervals
def calculate_prediction_intervals(model, X_test, y_test, alpha=0.05):
    # Make predictions
    y_pred = model.predict(X_test).flatten()

    # Calculate residuals
    residuals = y_test - y_pred

    # Estimate the standard deviation of the residuals
    std_dev = np.std(residuals)

    # Calculate the z-score for the desired confidence level
    z_score = 1.96  # For a 95% confidence interval

    # Calculate the margin of error
    margin_of_error = z_score * std_dev

    # Calculate lower and upper bounds of the prediction intervals
    lower_bound = y_pred - margin_of_error
    upper_bound = y_pred + margin_of_error

    return y_pred, lower_bound, upper_bound

def main():
    st.title("Stock Price Prediction with LSTM")

    # Create the models directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # User input for stock ticker
    ticker = st.text_input("Enter the stock ticker (e.g., AAPL):", "AAPL")

    # Check if the model already exists
    if os.path.exists(MODEL_FILE_PATH):
        model = load_model(MODEL_FILE_PATH)  # Load the existing model
        st.write("Loaded existing model.")
    else:
        # Fetch stock data
        data = fetch_stock_data(ticker)
        if data is None:
            st.error("Failed to fetch data. Please check the ticker symbol.")
            return

        # Filter to keep only the last 90 trading days
        data = data.tail(90)

        # Calculate features
        data = calculate_features(data)

        # Preprocess data
        data, features = preprocess_data(data)

        # Prepare data for LSTM
        window_size = 10  # You can change this value to experiment
        X, y, feature_scaler, target_scaler = prepare_data(data, features, window_size)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_lstm_model(X_train, y_train, X_test, y_test)

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

    # Display evaluation metrics
    st.subheader("Evaluation Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Plotting Actual vs Predicted with Prediction Intervals
    st.subheader("Actual vs Predicted Prices")
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
    if predicted_price is not None:
        st.write(f"The predicted closing price for the next trading day for {ticker} is: ${predicted_price:.2f}")

if __name__ == "__main__":
    main()
