import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import joblib # For saving and loading the scaler objects
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import streamlit as st

# Set a random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define the path for saving the model and scalers
MODEL_DIR = "models"
MODEL_FILENAME = "lstm_stock_predictor.keras"
FEATURE_SCALER_FILENAME = "feature_scaler.joblib"
TARGET_SCALER_FILENAME = "target_scaler.joblib"
WINDOW_SIZE = 15 # Increased window size for better LSTM performance
LOOKBACK_DAYS = 365 * 3 # Fetch 3 years of data to ensure indicators are calculated

# Create the model directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- 1. Data Retrieval Functions ---

@st.cache_data(show_spinner="Fetching stock data...")
def fetch_stock_data(ticker, lookback_days):
    """Function to fetch stock data for a specified lookback period."""
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=lookback_days) 
        
        # Use a longer period to ensure all EMAs/indicators are calculated
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if stock_data.empty:
            raise ValueError("No data found. Please check the ticker symbol or the time range.")
            
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {e}")
        return None

# --- 2. Technical Indicator Functions ---

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
    """Calculates all technical indicators and lagged features."""
    data['EMA_9'] = calculate_ema(data, 9)
    data['EMA_21'] = calculate_ema(data, 21)
    data['EMA_50'] = calculate_ema(data, 50)
    data['EMA_200'] = calculate_ema(data, 200) # Requires a long history!
    data['RSI'] = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_macd(data)

    data['Volume_Change'] = data['Volume'].pct_change()
    data['Price_Change'] = data['Close'].pct_change()
    
    # Lagged values for the last 5 days
    for lag in range(1, 6):
        data[f'Lag_Close_{lag}'] = data['Close'].shift(lag)
        data[f'Lag_Volume_{lag}'] = data['Volume'].shift(lag)
        data[f'Lag_EMA_9_{lag}'] = data['EMA_9'].shift(lag)
        data[f'Lag_RSI_{lag}'] = data['RSI'].shift(lag)
        data[f'Lag_MACD_{lag}'] = data['MACD'].shift(lag)
        data[f'Lag_Bollinger_Upper_{lag}'] = data['Bollinger_Upper'].shift(lag)
        data[f'Lag_Bollinger_Lower_{lag}'] = data['Bollinger_Lower'].shift(lag)

    # Drop rows with NaN values resulting from indicator/lag calculation
    # This keeps only the data points that have all required features
    data.dropna(inplace=True)
    return data

# --- 3. Preprocessing and Data Preparation Functions ---

def preprocess_data(data):
    """Adds the target variable and defines the feature list."""
    data['Target'] = data['Close'].shift(-1)
    data.dropna(subset=['Target'], inplace=True) # Drop the very last row which now has NaN target
    
    # Define the complete feature list
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
                'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'MACD', 'MACD_Signal', 'Volume_Change', 'Price_Change']
    
    # Add all lagged features
    for lag in range(1, 6):
        features.extend([
            f'Lag_Close_{lag}', f'Lag_Volume_{lag}', f'Lag_EMA_9_{lag}', f'Lag_RSI_{lag}',
            f'Lag_MACD_{lag}', f'Lag_Bollinger_Upper_{lag}', f'Lag_Bollinger_Lower_{lag}'
        ])

    return data, features

def prepare_data(data, features, window_size):
    """Scales data and creates LSTM sequences (X) and targets (y)."""
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(data[features])

    # Create sequences: X is the window_size sequence, y is the price immediately after
    X = np.array([scaled_features[i - window_size : i] 
                  for i in range(window_size, len(scaled_features))])
    
    # Target 'y' is the 'Target' column from the data frame, starting after the initial window_size
    y = data['Target'].values[window_size:]

    # Normalize the target variable
    y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    return X, y, feature_scaler, target_scaler

# --- 4. Model Training and Saving Functions ---

@st.cache_resource(show_spinner="Training LSTM model (this might take a few minutes)...")
def train_lstm_model(X_train, y_train, X_test, y_test, learning_rate=0.001, batch_size=32, epochs=100):
    """Builds, trains, and returns the LSTM model."""
    
    # Check if a trained model exists and load it
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if os.path.exists(model_path):
        st.info("Loading pre-trained model.")
        return load_model(model_path)
    
    st.info("No pre-trained model found. Training a new model.")
    
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.005), 
                           input_shape=(X_train.shape[1], X_train.shape[2]))),
        Dropout(0.3),
        Bidirectional(LSTM(64, kernel_regularizer=l2(0.005))),
        Dropout(0.3),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    # Callbacks to improve training stability and prevent overfitting
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=0 # Suppress detailed output during training
    )
    
    # Save the best model
    model.save(model_path)
    st.success(f"Model saved successfully to {model_path}.")

    return model

def save_scalers(feature_scaler, target_scaler):
    """Saves the feature and target scalers using joblib."""
    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, FEATURE_SCALER_FILENAME))
    joblib.dump(target_scaler, os.path.join(MODEL_DIR, TARGET_SCALER_FILENAME))

# --- 5. Prediction and Evaluation Functions ---

def predict_next_day(model, data, features, feature_scaler, target_scaler, window_size):
    """Predicts the next day's stock price."""
    
    # 1. Get the last 'window_size' rows of the feature data
    last_data = data[features].values[-window_size:]
    
    # 2. Scale the data
    last_data_scaled = feature_scaler.transform(last_data)
    
    # 3. Reshape for LSTM: (1, window_size, number_of_features)
    last_data_reshaped = last_data_scaled.reshape((1, window_size, len(features)))

    # 4. Predict and Inverse Transform
    predicted_price_scaled = model.predict(last_data_reshaped, verbose=0)[0][0]
    predicted_price = target_scaler.inverse_transform([[predicted_price_scaled]])[0][0]
    
    return predicted_price

def calculate_prediction_intervals(model, X_test, y_test, target_scaler, alpha=0.05):
    """Calculates a basic 95% confidence interval based on test residuals."""
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    residuals = y_test - y_pred_scaled
    std_dev = np.std(residuals)
    z_score = 1.96 # For a 95% confidence interval

    margin_of_error = z_score * std_dev

    lower_bound_scaled = y_pred_scaled - margin_of_error
    upper_bound_scaled = y_pred_scaled + margin_of_error
    
    # Inverse transform all
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    lower_bound = target_scaler.inverse_transform(lower_bound_scaled.reshape(-1, 1)).flatten()
    upper_bound = target_scaler.inverse_transform(upper_bound_scaled.reshape(-1, 1)).flatten()
    
    return y_pred, lower_bound, upper_bound

# --- 6. Streamlit Main Function ---

def main():
    st.set_page_config(page_title="LSTM Stock Price Predictor", layout="wide")
    st.title("📈 Stock Price Prediction using Bidirectional LSTM")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        ticker = st.text_input("Enter the stock ticker (e.g., AAPL):", "AAPL").upper()
        
        # User selection for training/testing period
        data_period_years = st.slider("Historical Data Period (Years)", 1, 5, 3)
        st.info(f"Fetching approximately {data_period_years} years of data.")
        
        test_size = st.slider("Test Set Size (%)", 5, 40, 20) / 100
        
        if st.button("Run Prediction & Training"):
            
            # --- Data Fetching and Preparation ---
            
            # Use the longer period based on user input
            lookback_days = data_period_years * 365 
            data = fetch_stock_data(ticker, lookback_days)
            
            if data is None:
                return

            st.success(f"Successfully fetched {len(data)} trading days of data for {ticker}.")

            # 1. Calculate features (requires long history)
            data = calculate_features(data)
            
            # 2. Preprocess data and define feature list
            data, features = preprocess_data(data)
            
            # 3. Check for sufficient data
            if len(data) < WINDOW_SIZE + 1:
                st.error(f"Error: Not enough data points ({len(data)}) after feature engineering. Need at least {WINDOW_SIZE + 1} for a window size of {WINDOW_SIZE}.")
                return

            # 4. Prepare data for LSTM (scaling and sequence creation)
            X, y, feature_scaler, target_scaler = prepare_data(data, features, WINDOW_SIZE)
            save_scalers(feature_scaler, target_scaler) # Save scalers for deployment

            # 5. Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False # Keep time-series order
            )

            # --- Model Training ---
            
            # The model is loaded/trained and saved within this function
            model = train_lstm_model(X_train, y_train, X_test, y_test)
            
            # --- Prediction and Evaluation ---
            
            # Calculate prediction intervals and predictions for the test set
            y_pred, lower_bound, upper_bound = calculate_prediction_intervals(
                model, X_test, y_test, target_scaler
            )
            
            # Inverse transform the actual test values for plotting and metrics
            y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # Predict the next day (unseen data)
            # Use the full, final processed data frame for the last sequence input
            predicted_price = predict_next_day(
                model, data, features, feature_scaler, target_scaler, WINDOW_SIZE
            )

            # --- Display Results ---
            
            st.subheader("Next Day Prediction")
            st.metric(label=f"Predicted Closing Price for {ticker}", value=f"${predicted_price:,.2f}")

            # Calculate performance metrics
            mae = mean_absolute_error(y_test_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
            mape = np.mean(np.abs((y_test_actual - y_pred) / (y_test_actual + 1e-8))) * 100

            st.subheader("Evaluation Metrics (on Test Set)")
            st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:,.2f}")
            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:,.2f}")
            st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{mape:,.2f}%")

            # Store results in session state for plotting
            st.session_state['plot_data'] = {
                'ticker': ticker,
                'dates': data.index[-len(y_test_actual):],
                'y_test_actual': y_test_actual,
                'y_pred': y_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

    with col2:
        st.subheader("Actual vs Predicted Prices")
        if 'plot_data' in st.session_state:
            plot_data = st.session_state['plot_data']
            
            fig = go.Figure()
            
            # Actual Prices
            fig.add_trace(go.Scatter(x=plot_data['dates'], y=plot_data['y_test_actual'], mode='lines', name='Actual Prices', line=dict(color='blue', width=2)))
            
            # Predicted Prices
            fig.add_trace(go.Scatter(x=plot_data['dates'], y=plot_data['y_pred'], mode='lines', name='Predicted Prices', line=dict(color='red', width=1)))
            
            # Prediction Interval (using fill for better visualization)
            fig.add_trace(go.Scatter(
                x=plot_data['dates'].tolist() + plot_data['dates'].tolist()[::-1],
                y=plot_data['upper_bound'].tolist() + plot_data['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='95% Prediction Interval'
            ))

            fig.update_layout(
                title=f'Actual vs Predicted Stock Prices for {plot_data["ticker"]} on Test Set',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click 'Run Prediction & Training' to generate the plot.")

if __name__ == "__main__":
    main()
