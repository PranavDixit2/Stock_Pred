import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential

# --- Feature calculation functions (omitted for brevity) ---

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
    data['BB_upper'] = sma + (2 * std_dev)
    data['BB_lower'] = sma - (2 * std_dev)
    return data

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['MACD_signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift(1))
    low_close = np.abs(data['Low'] - data['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()
    return atr

def calculate_features(data):
    data['EMA_9'] = calculate_ema(data, 9)
    data['EMA_21'] = calculate_ema(data, 21)
    data['EMA_50'] = calculate_ema(data, 50)
    data['EMA_200'] = calculate_ema(data, 200)
    data['RSI_14'] = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_macd(data)
    data['ATR_14'] = calculate_atr(data) 
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Price_Change'] = data['Close'].pct_change()
    lag_features = [
        'Close', 'Volume', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
        'RSI_14', 'ATR_14', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal',
        'Volume_Change', 'Price_Change'
    ]
    for feature in lag_features:
        for lag in range(1, 6):
            data[f'Lag_{feature}_{lag}'] = data[feature].shift(lag)
    data.dropna(inplace=True)
    return data

def preprocess_data(data):
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    target = log_returns.shift(-1)
    base_features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
        'RSI_14', 'ATR_14', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal', 
        'Volume_Change', 'Price_Change'
    ]
    lagged_feature_bases = [
        'Close', 'Volume', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
        'RSI_14', 'ATR_14', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal',
        'Volume_Change', 'Price_Change'
    ]
    features = base_features.copy()
    for feature_base in lagged_feature_bases:
        for lag in range(1, 6):
            features.append(f'Lag_{feature_base}_{lag}')
    target.dropna(inplace=True)
    data = data.loc[target.index] 
    return target, features

def prepare_data(data_df, features, target_series, window_size, feature_scaler, target_scaler):
    # Only scale and window the provided (sliced) data
    scaled_features = feature_scaler.transform(data_df[features])
    X = np.array([scaled_features[i-window_size:i] for i in range(window_size, len(scaled_features))])
    y_raw = target_series.values[window_size:]
    y = target_scaler.transform(y_raw.reshape(-1, 1)).flatten()
    return X, y

def fit_scalers(data_df, features, target_series):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(data_df[features])
    target_scaler.fit(target_series.values.reshape(-1, 1))
    return feature_scaler, target_scaler

def calculate_prediction_intervals(model, X_test, y_test, target_scaler, data_for_inversion):
    # Ensure all predictions are clean, 1D NumPy arrays
    y_pred_scaled = np.array(model.predict(X_test, verbose=0)).flatten()
    y_pred_log_return = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # If data_for_inversion has been correctly sliced to length N+1 (P_t and P_t+1):
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
    mae = mean_absolute_error(y_test_actual, y_pred)
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_actual - y_pred) / (y_test_actual + 1e-8))) * 100
    st.write("### Evaluation Metrics (on Test Set)")
    st.write(f"- Mean Absolute Error (MAE): ${mae:.2f}")
    st.write(f"- Root Mean Squared Error (RMSE): ${rmse:.2f}")
    st.write(f"- Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

def plot_predictions(data_for_lstm, y_test_actual, y_pred, lower_bound, upper_bound):
    test_dates = data_for_lstm.index[-len(y_test_actual):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_actual, mode='lines', name='Actual Prices', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred, mode='lines', name='Predicted Prices', line=dict(color='red', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=test_dates, y=upper_bound, mode='lines', name='Upper Bound (95% CI)', line=dict(color='rgba(128, 128, 128, 0.5)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=test_dates, y=lower_bound, mode='lines', name='95% Confidence Interval', line=dict(color='rgba(128, 128, 128, 0.5)', width=0), fill='tonexty', fillcolor='rgba(192, 192, 192, 0.3)'))
    fig.update_layout(title='B-LSTM T+1 Forecast Performance (Log Returns Target)', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

def predict_next_day(model, data_processed, features, feature_scaler, target_scaler, window_size):
    last_window_data = data_processed[features].values[-window_size:]
    last_window_scaled = feature_scaler.transform(last_window_data)
    last_window_reshaped = last_window_scaled.reshape((1, window_size, len(features)))
    predicted_log_return_scaled = model.predict(last_window_reshaped, verbose=0)
    predicted_log_return = target_scaler.inverse_transform(predicted_log_return_scaled).flatten()[0]
    close_t = data_processed['Close'].iloc[-1]
    predicted_price = close_t * np.exp(predicted_log_return)
    return float(predicted_price)

# --- Streamlit UI ---

st.title("Stock Price Prediction with Bidirectional LSTM (Log Returns)")
st.write("Predict next-day closing price with improved model architecture and log returns target.")

TICKER = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL").upper()
WINDOW_SIZE = st.sidebar.slider("LSTM Lookback Window", min_value=10, max_value=30, value=20) 
TRAIN_EPOCHS = 50

if st.button("Run Prediction (Final Fix)"):

    model_path = f"lstm_model_returns_{TICKER}.keras"
    scaler_path = f"scalers_returns_{TICKER}.joblib"

    with st.spinner("Fetching data..."):
        data_raw = yf.download(TICKER, period="5y", progress=False)
        if data_raw.empty:
            st.error("No data found for ticker. Please check the symbol.")
            st.stop()

    with st.spinner("Calculating features..."):
        data_features = calculate_features(data_raw.copy())
        if len(data_features) < 50:
            st.error(f"Not enough data after feature engineering ({len(data_features)} samples).")
            st.stop()

    with st.spinner("Preprocessing data..."):
        target, features = preprocess_data(data_features.copy()) 
        data_for_lstm = data_features.loc[target.index].copy()
        
        # Determine lengths for splitting
        N_samples = len(data_for_lstm)
        test_size_df = max(int(N_samples * 0.2), 10)
        
        # 1. SPLIT DATAFRAME FIRST (TRAIN & TEST)
        train_data_df = data_for_lstm.iloc[:-test_size_df]
        test_data_df = data_for_lstm.iloc[-test_size_df:]
        
        train_target = target.iloc[:-test_size_df]
        test_target = target.iloc[-test_size_df:]
        
        # 2. FIT SCALERS ON FULL TRAINING DATASET (BEFORE WINDOWING)
        feature_scaler, target_scaler = fit_scalers(data_for_lstm, features, target)

        # 3. CREATE WINDOWED ARRAYS (X_train, X_test, y_train, y_test)
        # We MUST ensure the window size is respected.
        if len(train_data_df) < WINDOW_SIZE:
             st.error(f"Training data ({len(train_data_df)} samples) is too short for window size ({WINDOW_SIZE}).")
             st.stop()
             
        # X_train and y_train
        X_train, y_train = prepare_data(train_data_df, features, train_target, WINDOW_SIZE, feature_scaler, target_scaler)
        
        # X_test and y_test: Need to include the end of the training data in the test features for the first window
        # Create a combined slice of (WINDOW_SIZE - 1) days of training data + test_data_df
        test_feature_window_start_index = len(data_for_lstm) - test_size_df - WINDOW_SIZE + 1
        
        # The data we need to create the test set X_test. Includes overlap from training set.
        data_for_X_test = data_for_lstm.iloc[test_feature_window_start_index:]
        
        # X_test and y_test
        X_test, y_test = prepare_data(data_for_X_test, features, target.iloc[test_feature_window_start_index:], WINDOW_SIZE, feature_scaler, target_scaler)


    if len(X_train) < 1:
        st.error(f"Insufficient samples ({len(X_train)}) after cleaning and windowing for training.")
        st.stop()

    N_test = len(y_test)
    
    # 4. SLICE INVERSION DATA (cleanly isolated)
    # This must be the last N_test + 1 days of the price data BEFORE the final prediction day.
    data_for_inversion = data_for_lstm[['Close']].iloc[-(N_test + 1):].copy()

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        feature_scaler, target_scaler = joblib.load(scaler_path)
        st.success("Loaded existing model and scalers.")
    else:
        st.info("Training new model...")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0005)), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32, kernel_regularizer=l2(0.0005))))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]

        history = model.fit(
            X_train, y_train,
            epochs=TRAIN_EPOCHS,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=callbacks_list,
            verbose=0
        )
        model.save(model_path)
        joblib.dump((feature_scaler, target_scaler), scaler_path)
        st.success("Training complete and model saved.")

    # Execute prediction and evaluation
    y_pred, lower_bound, upper_bound, y_test_actual = calculate_prediction_intervals(model, X_test, y_test, target_scaler, data_for_inversion)
    predicted_price = predict_next_day(model, data_for_lstm, features, feature_scaler, target_scaler, WINDOW_SIZE)

    st.write(f"### Predicted next-day closing price for {TICKER}: ${predicted_price:.2f}")

    display_evaluation_metrics(y_test_actual, y_pred)
    plot_predictions(data_for_lstm, y_test_actual, y_pred, lower_bound, upper_bound)
