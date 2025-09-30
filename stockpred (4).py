import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA # <-- NEW: For Feature Reduction
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, BatchNormalization
from tensorflow.keras.regularizers import l2
from kerastuner import HyperModel, RandomSearch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Feature & Data Functions ---

def fetch_stock_data(ticker):
    data_raw = yf.download(ticker)
    if data_raw.empty:
        st.error("No data found for ticker. Please check the symbol.")
        st.stop()
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
        
        # Reduced max units to 128
        model.add(Bidirectional(GRU(hp.Int('units_1', min_value=32, max_value=128, step=32), 
                                    return_sequences=True), input_shape=(None, self.feature_count)))
        model.add(BatchNormalization())
        # Reduced max dropout to 0.3
        model.add(Dropout(hp.Float('dropout_1', 0.1, 0.3, step=0.1)))
        
        # Reduced max units to 128
        model.add(Bidirectional(GRU(hp.Int('units_2', min_value=32, max_value=128, step=32))))
        model.add(BatchNormalization())
        # Reduced max dropout to 0.3
        model.add(Dropout(hp.Float('dropout_2', 0.1, 0.3, step=0.1)))
        
        model.add(Dense(1)) 
        
        # IMPROVEMENT: Using Huber Loss to penalize outliers less
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-3, sampling='LOG')),
                      loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
        return model

def train_model_with_tuning(X_train, y_train, X_test, y_test, feature_count):
    # Increased max_trials for deeper search
    tuner = RandomSearch(LSTMHyperModel(feature_count).build, 
                         objective='val_mae', 
                         max_trials=10, 
                         executions_per_trial=1,
                         directory='kt_dir',
                         project_name='stock_pred_huber') # New project name
    
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

    # Increased epochs for better training
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks_list, verbose=0)
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

# --- Data Preparation & Prediction Functions ---

def prepare_data(data, features, window_size=20, pca_variance=0.85): # Added PCA variance threshold
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(data[features])
    
    # NEW: Apply PCA for Dimensionality Reduction 
    pca = PCA(n_components=pca_variance) 
    pca.fit(scaled_features)
    scaled_features_reduced = pca.transform(scaled_features)
    
    st.write(f"Feature count reduced from **{len(features)}** to **{scaled_features_reduced.shape[1]}** (retaining {pca_variance*100:.0f}% variance).")

    # Create time series windows (X)
    X = np.array([scaled_features_reduced[i-window_size:i] for i in range(window_size, len(scaled_features_reduced))])
    
    # Target variable (y)
    y_raw = data['Target'].values[window_size:]
    y = target_scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()
    
    # Return the PCA object along with scalers
    return X, y, feature_scaler, target_scaler, pca

def predict_next_day(model, data, features, feature_scaler, target_scaler, pca, window_size=20): # Added PCA object
    last_data = data[features].values[-window_size:]
    last_data_scaled = feature_scaler.transform(last_data)
    
    # NEW: Transform feature data using the fitted PCA object
    last_data_scaled_reduced = pca.transform(last_data_scaled) 
    
    last_data_reshaped = last_data_scaled_reduced.reshape((1, window_size, last_data_scaled_reduced.shape[1]))

    predicted_change_scaled = model.predict(last_data_reshaped, verbose=0)[0][0]
    
    predicted_change = target_scaler.inverse_transform([[predicted_change_scaled]])[0][0]
    
    current_price = data['Close'].values[-1]
    predicted_price = current_price * (1 + predicted_change)
    
    return float(predicted_price)

# --- Evaluation Functions (Metrics function uses safe arrays) ---

def calculate_prediction_intervals(model, X_test, y_test, target_scaler, data_for_inversion):
    y_pred_scaled = np.array(model.predict(X_test, verbose=0)).flatten()
    y_pred_log_return = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    close_t = data_for_inversion['Close'].values[:-1]
    y_test_actual = data_for_inversion['Close'].values[1:]
    
    y_pred_actual = close_t * np.exp(y_pred_log_return) 

    residuals = y_test_actual - y_pred_actual
    std_dev = np.std(residuals)
    z_score = 1.96
    margin_of_error = z_score * std_dev
    
    lower_bound = (y_pred_actual - margin_of_error).flatten()
    upper_bound = (y_pred_actual + margin_of_error).flatten()
    
    return y_pred_actual.flatten(), lower_bound, upper_bound, y_test_actual.flatten()

def display_evaluation_metrics(y_test_actual, y_pred):
    y_true_safe = np.array(y_test_actual)
    y_pred_safe = np.array(y_pred)
    
    if len(y_true_safe) != len(y_pred_safe):
        st.error(f"FATAL ERROR: Inconsistent sample size in evaluation. Actual: {len(y_true_safe)}, Predicted: {len(y_pred_safe)}. Check data splitting logic.")
        raise ValueError("Inconsistent numbers of samples found in evaluation inputs.")

    mae = mean_absolute_error(y_true_safe, y_pred_safe)
    mse = mean_squared_error(y_true_safe, y_pred_safe)
    rmse = np.sqrt(mse)
    
    mape = np.mean(np.abs((y_true_safe - y_pred_safe) / (y_true_safe + 1e-8))) * 100
    
    st.write("### Evaluation Metrics (on Test Set - Percentage Change)")
    st.write(f"- Mean Absolute Error (MAE): **{mae:.4f}**")
    st.write(f"- Root Mean Squared Error (RMSE): {rmse:.4f}")
    st.write(f"- Mean Absolute Percentage Error (MAPE): {mape:.2f}% (Ignored due to near-zero targets)")
    
# --- Streamlit Application ---

def main():
    st.title("Stock Price Prediction with B-GRU LSTM (PCA & Huber Loss)")
    st.write("Enhanced training with feature reduction and improved outlier handling.")

    TICKER = st.text_input("Enter the stock ticker (e.g., AAPL):", "AAPL").upper()
    WINDOW_SIZE = st.sidebar.slider("LSTM Lookback Window", min_value=10, max_value=30, value=20)
    TEST_SIZE = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=30, value=20) / 100
    PCA_VARIANCE = st.sidebar.slider("PCA Variance Threshold", min_value=0.70, max_value=0.95, value=0.85, step=0.05)

    if st.button("Run Prediction"):
        
        with st.spinner("Fetching and preparing data..."):
            data = fetch_stock_data(TICKER)
            
            data = data.tail(1250) # Use the last 5 years of data
            
            data = calculate_features(data.copy())
            data, features = preprocess_data(data.copy())

            # Prepare data for LSTM (now includes PCA)
            X, y, feature_scaler, target_scaler, pca = prepare_data(data, features, WINDOW_SIZE, PCA_VARIANCE)
            
            if len(X) < 50:
                 st.error(f"Insufficient samples ({len(X)}) after windowing and cleaning. Need at least 50.")
                 st.stop()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, shuffle=False, random_state=None
            )
            feature_count = X_train.shape[2] # Use the reduced feature count

        with st.spinner("Training model with hyperparameter tuning..."):
            model = train_model_with_tuning(X_train, y_train, X_test, y_test, feature_count)
            st.success("Training and tuning complete.")

        # --- Evaluation and Output ---
        with st.spinner("Calculating predictions and metrics..."):
            
            y_pred_scaled = model.predict(X_test, verbose=0).flatten()
            residuals_scaled = y_test - y_pred_scaled
            
            lower_bound_scaled = y_pred_scaled + np.percentile(residuals_scaled, 2.5)
            upper_bound_scaled = y_pred_scaled + np.percentile(residuals_scaled, 97.5)
            
            # Inverse transform ALL scaled results
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            lower_bound = target_scaler.inverse_transform(lower_bound_scaled.reshape(-1, 1)).flatten()
            upper_bound = target_scaler.inverse_transform(upper_bound_scaled.reshape(-1, 1)).flatten()

            # The evaluation metrics now use the correct length and content (actual percentage change)
            display_evaluation_metrics(y_test_actual, y_pred)
            
            predicted_price = predict_next_day(model, data, features, feature_scaler, target_scaler, pca, WINDOW_SIZE)
            st.write(f"### Predicted next-day closing price for {TICKER}: **${predicted_price:.2f}**")

            # --- Plotting Price Levels ---
            N_test = len(y_test)
            
            start_price_index = len(X) - len(X_test) - 1 
            start_price = data['Close'].iloc[start_price_index]
            start_price_scalar = float(start_price) # Convert to scalar float
            
            price_base = np.insert(y_test_actual, 0, 0.0)
            predicted_base = np.insert(y_pred, 0, 0.0)
            
            actual_prices_full = start_price_scalar * (1 + price_base).cumprod()
            predicted_prices_full = start_price_scalar * (1 + predicted_base).cumprod()
            
            actual_prices = actual_prices_full[1:] 
            predicted_prices = predicted_prices_full[1:] 
            
            lower_bound_prices = predicted_prices + (lower_bound * start_price_scalar) 
            upper_bound_prices = predicted_prices + (upper_bound * start_price_scalar) 

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
    main()
