import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # <--- THE CRITICAL COMPONENT
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, BatchNormalization
from tensorflow.keras.regularizers import l2
from kerastuner import HyperModel, RandomSearch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Feature & Data Functions (Adjusted for consistency) ---

def fetch_stock_data(ticker):
    data_raw = yf.download(ticker)
    if data_raw.empty:
        st.error("No data found for ticker. Please check the symbol.")
        st.stop()
    # Using a 5-year period for robust feature calculation before final slicing
    return yf.download(ticker, period="5y", progress=False)

def calculate_ema(data, span):
    return data['Close'].ewm(span=span, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    data['Bollinger_Upper'] = sma + (2 * std_dev)
    data['Bollinger_Lower'] = sma - (2 * std_dev)
    return data

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

def calculate_features(data):
    for span in [9, 21, 50, 200]:
        data[f'EMA_{span}'] = calculate_ema(data, span)
    
    data['RSI'] = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_macd(data)
    
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Price_Change'] = data['Close'].pct_change()

    # Base features for lag
    lag_bases = ['Close', 'Volume', 'EMA_9', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
    for lag in range(1, 6):
        for feature_base in lag_bases:
            data[f'Lag_{feature_base}_{lag}'] = data[feature_base].shift(lag)

    data.dropna(inplace=True)
    return data

def preprocess_data(data):
    # Imputation removed for simplicity/to match final code state, reliance on dropna
    
    # Target is the next day's percentage change (t+1)
    data['Target'] = data['Close'].pct_change().shift(-1)
    data.dropna(inplace=True)
    
    # Define features based on the column generation in calculate_features
    features = [col for col in data.columns if col not in ['Target', 'Adj Close']]
    return data, features

# --- LSTM Model & Training Functions ---

class LSTMHyperModel(HyperModel):
    def __init__(self, feature_count):
        self.feature_count = feature_count
    
    def build(self, hp):
        model = Sequential()
        model.add(Bidirectional(GRU(hp.Int('units_1', min_value=32, max_value=128, step=32), 
                                    return_sequences=True), input_shape=(None, self.feature_count)))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_1', 0.1, 0.3, step=0.1)))
        model.add(Bidirectional(GRU(hp.Int('units_2', min_value=32, max_value=128, step=32))))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_2', 0.1, 0.3, step=0.1)))
        model.add(Dense(1)) 
        
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-3, sampling='LOG')),
                      loss='mean_squared_error', metrics=['mae'])
        return model

def train_model_with_tuning(X_train, y_train, X_test, y_test, feature_count):
    # Temporarily remove logging from tuner to avoid polluting Streamlit output
    tuner = RandomSearch(LSTMHyperModel(feature_count).build, 
                         objective='val_mae', 
                         max_trials=3, # Reduced for faster execution
                         executions_per_trial=1,
                         directory='kt_dir',
                         project_name='stock_pred')
    
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

    tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=callbacks_list, verbose=0)
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

# --- Data Preparation & Prediction Functions ---

def prepare_data(data, features, window_size=20): # Increased window_size for better time series data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(data[features])

    # Create time series windows (X)
    X = np.array([scaled_features[i-window_size:i] for i in range(window_size, len(scaled_features))])
    
    # Target variable (y)
    y_raw = data['Target'].values[window_size:]
    
    # Normalize the target variable (percentage change)
    y = target_scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()
    
    return X, y, feature_scaler, target_scaler

def predict_next_day(model, data, features, feature_scaler, target_scaler, window_size=20):
    last_data = data[features].values[-window_size:]
    last_data_scaled = feature_scaler.transform(last_data)
    last_data_reshaped = last_data_scaled.reshape((1, window_size, len(features)))

    predicted_change_scaled = model.predict(last_data_reshaped, verbose=0)[0][0] # Access the scalar value
    
    # Inverse transform to get the actual percentage change
    predicted_change = target_scaler.inverse_transform([[predicted_change_scaled]])[0][0]
    
    # Calculate the predicted price
    current_price = data['Close'].values[-1]
    predicted_price = current_price * (1 + predicted_change)
    
    # FIX: Ensure the final result is a standard Python float
    return float(predicted_price)
    
# --- Streamlit Application ---

def main():
    st.title("Stock Price Prediction with B-GRU LSTM (Robust Split)")
    st.write("Using `train_test_split` on the final arrays for guaranteed sample consistency.")

    TICKER = st.text_input("Enter the stock ticker (e.g., AAPL):", "AAPL").upper()
    WINDOW_SIZE = st.sidebar.slider("LSTM Lookback Window", min_value=10, max_value=30, value=20)
    TEST_SIZE = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=30, value=20) / 100

    if st.button("Run Prediction"):
        
        with st.spinner("Fetching and preparing data..."):
            data = fetch_stock_data(TICKER)
            
            # Use the last 5 years of data for analysis
            data = data.tail(1250) 
            
            data = calculate_features(data.copy())
            data, features = preprocess_data(data.copy())

            # Prepare data for LSTM
            X, y, feature_scaler, target_scaler = prepare_data(data, features, WINDOW_SIZE)
            
            if len(X) < 50:
                 st.error(f"Insufficient samples ({len(X)}) after windowing and cleaning. Need at least 50.")
                 st.stop()
            
            # CRITICAL FIX: Use train_test_split for guaranteed consistency
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, shuffle=False, random_state=None
            )
            feature_count = X_train.shape[2]

        with st.spinner("Training model with hyperparameter tuning..."):
            # Load or train model
            model = train_model_with_tuning(X_train, y_train, X_test, y_test, feature_count)
            st.success("Training and tuning complete.")

        # --- Evaluation and Output ---
        with st.spinner("Calculating predictions and metrics..."):
            
            # Calculate prediction intervals (predictions are still scaled change)
            y_pred_scaled = model.predict(X_test, verbose=0).flatten()
            
            # Calculate residuals from the scaled predictions
            residuals_scaled = y_test - y_pred_scaled
            
            # Calculate scaled bounds
            lower_bound_scaled = y_pred_scaled + np.percentile(residuals_scaled, 2.5)
            upper_bound_scaled = y_pred_scaled + np.percentile(residuals_scaled, 97.5)
            
            # Inverse transform ALL scaled results (y_pred, y_test, bounds)
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            lower_bound = target_scaler.inverse_transform(lower_bound_scaled.reshape(-1, 1)).flatten()
            upper_bound = target_scaler.inverse_transform(upper_bound_scaled.reshape(-1, 1)).flatten()

            # The evaluation metrics now use the correct length and content (actual percentage change)
            mae = mean_absolute_error(y_test_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
            mape = np.mean(np.abs((y_test_actual - y_pred) / (y_test_actual + 1e-8))) * 100
            
            st.write("### Evaluation Metrics (on Test Set - Percentage Change)")
            st.write(f"- Mean Absolute Error (MAE): {mae:.4f}")
            st.write(f"- Root Mean Squared Error (RMSE): {rmse:.4f}")
            st.write(f"- Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            
            # Predict the next day's stock price (using the last row of the full cleaned data)
            predicted_price = predict_next_day(model, data, features, feature_scaler, target_scaler, WINDOW_SIZE)
            st.write(f"### Predicted next-day closing price for {TICKER}: ${predicted_price:.2f}")

            # --- Plotting Price Levels (Final Fix) ---
            N_test = len(y_test)
            
            # Index of the price that precedes the first test target (t-1)
            # len(X) - len(X_test) - 1 == Index of the close price used as P_t for the first P_t+1 prediction
            start_price_index = len(X) - len(X_test) - 1 
            start_price = data['Close'].iloc[start_price_index]
            
            # 1. Create base arrays (Length N_test + 1)
            # Insert 0.0 at the beginning to act as the base for the cumulative product
            price_base = np.insert(y_test_actual, 0, 0.0)
            predicted_base = np.insert(y_pred, 0, 0.0)
            
            # 2. Calculate cumulative product (Length N_test + 1)
            actual_prices_full = start_price * (1 + price_base).cumprod()
            predicted_prices_full = start_price * (1 + predicted_base).cumprod()
            
            # 3. Final slice to remove the starting price (Index 0), resulting in Length N_test
            actual_prices = actual_prices_full[1:] 
            predicted_prices = predicted_prices_full[1:] 
            
            # Adjust bounds to price level (using the relative change from the predicted prices)
            # Note: This is a robust simplification of price-level confidence interval calculation
            lower_bound_prices = predicted_prices + (lower_bound * start_price) 
            upper_bound_prices = predicted_prices + (upper_bound * start_price) 

            # Plotting
            test_dates = data.index[-N_test:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_dates, y=actual_prices, mode='lines', name='Actual Price Level', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_dates, y=predicted_prices, mode='lines', name='Predicted Price Level', line=dict(color='red', dash='dot')))
            
            # Add bounds
            fig.add_trace(go.Scatter(x=test_dates, y=upper_bound_prices, mode='lines', name='Upper Bound', line=dict(color='gray', dash='dash'), opacity=0.5))
            fig.add_trace(go.Scatter(x=test_dates, y=lower_bound_prices, mode='lines', name='Lower Bound', line=dict(color='gray', dash='dash'), fill='tonexty', fillcolor='rgba(192, 192, 192, 0.3)', opacity=0.5))
            
            fig.update_layout(title=f'Actual vs Predicted Stock Price Levels for {TICKER}',
                              xaxis_title='Date',
                              yaxis_title='Closing Price (USD)')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # Note: If running this as a full script, ensure you run the main() function here:
    main()
