import os 
import yfinance as yf 
import pandas as pd 
import numpy as np 
import plotly.graph_objects as go 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional 
from tensorflow.keras.regularizers import l2 
import tensorflow as tf 
import streamlit as st 
import joblib 
import warnings 
from sklearn.model_selection import train_test_split 

# Suppress TensorFlow warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
# Note: os.environ assignment is kept based on previous context, but may not be strictly necessary
os.environ = '2' 

# --- I. DATA ACQUISITION AND FEATURE ENGINEERING --- 

def fetch_stock_data(ticker): 
    """Fetches historical stock data using yfinance."""
    try: 
        # Fetch maximum available history for stable indicator calculation
        stock_data = yf.download(ticker) 
        if stock_data.empty: 
            raise ValueError("No data found. Please check the ticker symbol.") 
        st.info(f"Successfully fetched {len(stock_data)} days of data for {ticker}.")
        return stock_data 
    except Exception as e: 
        st.error(f"Error fetching stock data: {e}") 
        return None 

def calculate_ema(data, span): 
    """Calculates Exponential Moving Average (EMA)."""
    return data['Close'].ewm(span=span, adjust=False).mean() 

def calculate_rsi(data, window=14): 
    """Calculates Relative Strength Index (RSI)."""
    delta = data['Close'].diff() 
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean() 
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean() 
    rs = gain / (loss + 1e-8)  # Prevent division by zero 
    rsi = 100 - (100 / (1 + rs)) 
    return rsi 

def calculate_bollinger_bands(data, window=20): 
    """Calculates Bollinger Bands (2 std deviations) and adds to DataFrame."""
    sma = data['Close'].rolling(window=window).mean() 
    std_dev = data['Close'].rolling(window=window).std() 
    # Corrected assignment to update the DataFrame columns
    data = sma + (2 * std_dev) 
    data = sma - (2 * std_dev) 
    return data

def calculate_macd(data, short_window=12, long_window=26, signal_window=9): 
    """Calculates Moving Average Convergence Divergence (MACD) and adds to DataFrame."""
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean() 
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean() 
    # Corrected assignment to update the DataFrame columns
    data = short_ema - long_ema 
    data = data.ewm(span=signal_window, adjust=False).mean() 
    return data 

def calculate_features(data): 
    """
    Calculates 5 key technical indicators and extensive lagged features.
    """
    # 1. Trend Indicators (EMA variants)
    data['EMA_9'] = calculate_ema(data, 9) 
    data['EMA_21'] = calculate_ema(data, 21) 
    data['EMA_50'] = calculate_ema(data, 50) 
    data['EMA_200'] = calculate_ema(data, 200) 
    
    # 2. Momentum Indicator
    data = calculate_rsi(data) 
    
    # 3. Volatility Indicator (Updates data in place and returns it)
    data = calculate_bollinger_bands(data) 
    
    # 4. Convergence/Divergence Indicator (Updates data in place and returns it)
    data = calculate_macd(data) 

    # 5. Volume and Price Rate of Change
    data['Volume_Change'] = data['Volume'].pct_change() 
    data['Price_Change'] = data['Close'].pct_change() 

    # 6. Sequential Memory Features (Lagged Values - 5 days) 
    # FIX: Completed list assignment (addresses SyntaxError)
    lag_features =
    for feature in lag_features:
         for lag in range(1, 6):
             data[f'Lag_{feature}_{lag}'] = data[feature].shift(lag)

    # Drop initial rows with NaN values resulting from indicator and lag calculations
    data.dropna(inplace=True) 
    st.info(f"Data cleaning complete. {len(data)} samples remaining for training/testing.")
    return data 

def preprocess_data(data): 
    """Defines the target variable and lists the final feature set."""
    # Target: Predict the next day's closing price
    data = data['Close'].shift(-1) 
    data.dropna(inplace=True) 
    
    # Final comprehensive feature list
    # FIX: Completed list assignment for all core non-lagged features
    base_features =
    
    # Features that were lagged in calculate_features
    # FIX: Completed list assignment for features to be lagged
    lagged_feature_bases =
    
    features = base_features.copy()
    
    # Dynamically add the 35 lagged columns to the feature list
    for feature_base in lagged_feature_bases:
        for lag in range(1, 6):
            features.append(f'Lag_{feature_base}_{lag}')

    # Return the target column Series and the final list of features
    return data, features 

# --- II. LSTM DATA PREPARATION AND TRAINING --- 

def prepare_data(data, features, window_size=10): 
    """Normalizes data and converts it into the 3D tensor format for LSTM input."""
    feature_scaler = MinMaxScaler() 
    target_scaler = MinMaxScaler() 

    # 1. Scale Features
    scaled_features = feature_scaler.fit_transform(data[features]) 

    # 2. Create Sequences (X)
    # X: (Samples, Time Steps (Window Size), Features)
    X = np.array([scaled_features[i-window_size:i] 
                  for i in range(window_size, len(scaled_features))]) 
    
    # 3. Scale Target (y)
    y_raw = data.values[window_size:] # Use the defined 'Target' column
    y = target_scaler.fit_transform(y_raw.reshape(-1, 1)).flatten() 

    return X, y, feature_scaler, target_scaler 

def train_lstm_model(X_train, y_train, X_test, y_test, learning_rate=0.001, batch_size=64, epochs=50): 
    """Builds and trains the Bidirectional LSTM model with regularization and callbacks."""
    
    # Architectural configuration
    # Correctly extracting the input shape: (Timesteps, Num_Features)
    input_shape = (X_train.shape, X_train.shape) 
    
    # FIX: Restored the model definition
    model = Sequential() 

    # Optimization strategy
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='mean_squared_error', 
                  metrics=['mae']) 
    
    st.info("Starting model training (B-LSTM)...")
    
    # Training protocol with adaptive callbacks
    # FIX: Restored the full callback list
    callbacks_list =

    history = model.fit( 
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test), 
        callbacks=callbacks_list,
        verbose=0 
    ) 
    st.success(f"Training finished. Model restored to best weights based on validation loss.")
    return model 

# --- III. MODEL PERSISTENCE AND INFERENCE --- 

def save_model_and_scalers(model, feature_scaler, target_scaler, ticker): 
    """Saves the trained model and normalization scalers for future use."""
    MODEL_DIR = 'models'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, f'lstm_model_{ticker}.keras') 
    scaler_path = os.path.join(MODEL_DIR, f'scalers_{ticker}.joblib') 
    
    # 1. Save Model (Keras native format)
    model.save(model_path) 
    
    # 2. Save Scalers (using joblib)
    joblib.dump((feature_scaler, target_scaler), scaler_path) 
    st.success(f"Model and scalers saved successfully for {ticker}.") 

def load_model_and_scalers(ticker): 
    """Loads existing model and scalers if available."""
    MODEL_DIR = 'models'
    model_path = os.path.join(MODEL_DIR, f'lstm_model_{ticker}.keras') 
    scaler_path = os.path.join(MODEL_DIR, f'scalers_{ticker}.joblib') 
    
    if os.path.exists(model_path) and os.path.exists(scaler_path): 
        try: 
            # Load model
            model = tf.keras.models.load_model(model_path) 
            
            # Load scalers
            feature_scaler, target_scaler = joblib.load(scaler_path) 
            st.info(f"Loaded existing model and scalers for {ticker}. Skipping training.") 
            return model, feature_scaler, target_scaler 
        except Exception as e: 
            st.error(f"Error loading model/scalers: {e}. Retraining required.") 
            return None, None, None 
    else: 
        st.info(f"No saved model found for {ticker}. Training new model...") 
        return None, None, None 

def predict_next_day(model, data_processed, features, feature_scaler, target_scaler, window_size=10): 
    """
    Predicts the next day's stock price using the last sequence of available data.
    """
    # 1. Get the last required sequence (10 days)
    # The data_processed input here is the DataFrame *after* all features and target columns were calculated
    # but *before* the target shift caused the last row (which contains NaN for target) to be dropped.
    # We must use the last valid row from the original feature calculation set.
    last_data = data_processed[features].values[-window_size:] 
    
    # 2. Scale the sequence using the saved feature_scaler
    last_data_scaled = feature_scaler.transform(last_data) 
    
    # 3. Reshape for LSTM input: (1, Timesteps, Features)
    last_data_reshaped = last_data_scaled.reshape((1, window_size, len(features))) 

    # 4. Predict (output is scaled)
    predicted_price_scaled = model.predict(last_data_reshaped, verbose=0) 
    
    # 5. Inverse transform to actual dollar price
    predicted_price = target_scaler.inverse_transform(predicted_price_scaled) # Correct indexing
    return predicted_price 

# --- IV. EVALUATION AND UNCERTAINTY QUANTIFICATION --- 

def calculate_prediction_intervals(model, X_test, y_test, target_scaler): 
    """
    Calculates 95% Prediction Intervals based on test set residuals (assuming normality).
    """
    # 1. Make scaled predictions
    y_pred_scaled = model.predict(X_test, verbose=0).flatten() 

    # 2. Inverse transform all values for practical metric calculation
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten() 
    y_pred_actual = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten() 
    
    # 3. Calculate residuals (in actual dollar values)
    residuals = y_test_actual - y_pred_actual 

    # 4. Estimate the standard deviation of the residuals (sigma_res)
    std_dev = np.std(residuals) 

    # 5. Calculate 95% confidence bounds (Z-score = 1.96)
    z_score = 1.96  
    margin_of_error = z_score * std_dev 

    lower_bound = y_pred_actual - margin_of_error 
    upper_bound = y_pred_actual + margin_of_error 
    
    return y_pred_actual, lower_bound, upper_bound, y_test_actual 

def display_evaluation_metrics(y_test_actual, y_pred):
    """Calculates and displays standard regression metrics."""
    mae = mean_absolute_error(y_test_actual, y_pred) 
    mse = mean_squared_error(y_test_actual, y_pred) 
    rmse = np.sqrt(mse) 
    # Handle potential division by zero for MAPE calculation
    mape = np.mean(np.abs((y_test_actual - y_pred) / (y_test_actual + 1e-8))) * 100 

    st.subheader("Evaluation Metrics (on Test Set)") 
    st.markdown(f"**Mean Absolute Error (MAE):** ${mae:.2f} (Average dollar error)")
    st.markdown(f"**Root Mean Squared Error (RMSE):** ${rmse:.2f} (Penalizes large errors)") 
    st.markdown(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}% (Scale-independent performance)") 

def plot_predictions(data, ticker, y_test_actual, y_pred, lower_bound, upper_bound):
    """Generates a Plotly visualization of performance and uncertainty."""
    st.subheader("Actual vs Predicted Prices with 95% Prediction Intervals") 
    
    # Select the dates corresponding to the test set 
    test_dates = data.index[-len(y_test_actual):]

    fig = go.Figure() 
    
    # Actual Prices
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_actual, mode='lines', 
                             name='Actual Prices', line=dict(color='blue', width=2))) 
    
    # Predicted Prices
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred, mode='lines', 
                             name='Predicted Prices', line=dict(color='red', width=2, dash='dot'))) 
    
    # Prediction Intervals (Uncertainty Band)
    fig.add_trace(go.Scatter(x=test_dates, y=upper_bound, mode='lines', 
                             name='Upper Bound (95% CI)', 
                             line=dict(color='rgba(128, 128, 128, 0.5)', width=0), 
                             showlegend=False)) 
    
    fig.add_trace(go.Scatter(x=test_dates, y=lower_bound, mode='lines', 
                             name='95% Confidence Interval', 
                             line=dict(color='rgba(128, 128, 128, 0.5)', width=0), 
                             fill='tonexty', fillcolor='rgba(192, 192, 192, 0.3)')) 
    
    fig.update_layout(title=f'B-LSTM T+1 Forecast Performance for {ticker}', 
                      xaxis_title='Date', 
                      yaxis_title='Price (USD)',
                      template='plotly_white') 
    st.plotly_chart(fig, use_container_width=True) 

# --- V. MAIN EXECUTION FLOW --- 

def main(): 
    st.set_page_config(layout="wide")
    st.title("Expert-Level Stock Price Prediction Framework (B-LSTM)") 

    # --- Constants/Hyperparameters --- 
    WINDOW_SIZE = 10 
    TRAIN_EPOCHS = 50 
    
    ticker = st.text_input("Enter the stock ticker (e.g., AAPL):", "AAPL").upper() 
    
    if not ticker: 
        st.warning("Please enter a stock ticker.") 
        return 

    # 1. Fetch data
    data_raw = fetch_stock_data(ticker) 
    if data_raw is None: 
        return 

    # 2. Calculate features (on full history)
    data_features = data_raw.copy()
    data_features = calculate_features(data_features) 
    
    # 3. Handle Persistence (Attempt to load before proceeding to training/scaling)
    model, feature_scaler, target_scaler = load_model_and_scalers(ticker) 

    # 4. Preprocess data (Defines Target, calculates final features list)
    # The 'data' passed here needs to have all features calculated for the splitting logic.
    # We pass data_features to preprocess_data which will extract the target and features list
    data_target, features = preprocess_data(data_features.copy()) 
    
    # The data_for_lstm is the DataFrame used for sequence generation, which must have
    # the target column appended for the next step.
    data_for_lstm = data_features.loc[data_target.index]
    data_for_lstm = data_target
    
    # 5. Prepare data for sequence modeling
    X, y, initial_feature_scaler, initial_target_scaler = prepare_data(data_for_lstm, features, WINDOW_SIZE) 
    
    if len(X) < 30: # Minimum reasonable length for split
        st.error(f"Insufficient samples ({len(X)}) after cleaning and windowing. Required at least 30 samples for meaningful training.")
        return

    # Split the data chronologically (80% training, 20% testing)
    test_size = max(int(len(X) * 0.2), 10) 
    X_train = X[:-test_size] 
    X_test = X[-test_size:] 
    y_train = y[:-test_size] 
    y_test = y[-test_size:] 

    # 6. Training Logic
    if model is None: 
        # Use the newly fitted scalers if training
        feature_scaler = initial_feature_scaler
        target_scaler = initial_target_scaler
        
        # Train and save the new model
        model = train_lstm_model(X_train, y_train, X_test, y_test, epochs=TRAIN_EPOCHS) 
        save_model_and_scalers(model, feature_scaler, target_scaler, ticker) 
    elif feature_scaler is None or target_scaler is None:
        # If model loaded but scalers failed, use the newly fitted scalers from preparation
        feature_scaler = initial_feature_scaler
        target_scaler = initial_target_scaler
        st.warning("Model loaded, but scalers were missing/corrupted. Using newly fitted scalers.")

    # 7. Evaluation and Prediction
    try: 
        # Calculate prediction intervals and inverse transform all data
        y_pred, lower_bound, upper_bound, y_test_actual = calculate_prediction_intervals(
            model, X_test, y_test, target_scaler) 

        # Predict the next day's stock price (T+1 Inference)
        # Use the data_features DF (which has all features calculated up to the latest date)
        predicted_price = predict_next_day(model, data_features, features, feature_scaler, target_scaler, WINDOW_SIZE) 
        
        st.subheader(f"T+1 Prediction for {ticker}")
        st.metric(label="Predicted Next Day Close Price", 
                  value=f"${predicted_price:.2f}")

        # Display performance metrics
        display_evaluation_metrics(y_test_actual, y_pred)

        # Plot results (use the data_for_lstm which contains the dates corresponding to the test set)
        plot_predictions(data_for_lstm, ticker, y_test_actual, y_pred, lower_bound, upper_bound)
        
    except Exception as e: 
        st.error(f"A critical error occurred during prediction or evaluation: {e}") 

if __name__ == "__main__": 
    main()
