    # Try loading model and scalers from local or GitHub (you can customize this)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        feature_scaler, target_scaler = joblib.load(scaler_path)
        st.success("Loaded existing model and scalers.")
    else:
        st.info("Training new model...")
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.models import Sequential
        import tensorflow as tf
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32, kernel_regularizer=l2(0.001))))
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
    # Evaluation and prediction
    y_pred, lower_bound, upper_bound, y_test_actual = calculate_prediction_intervals(model, X_test, y_test, target_scaler)
    predicted_price = predict_next_day(model, data_features, features, feature_scaler, target_scaler, WINDOW_SIZE)
    st.write(f"### Predicted next-day closing price for {TICKER}: ${predicted_price:.2f}")
    display_evaluation_metrics(y_test_actual, y_pred)
    plot_predictions(data_for_lstm, y_test_actual, y_pred, lower_bound, upper_bound)
