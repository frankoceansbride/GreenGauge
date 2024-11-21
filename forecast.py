import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the LSTM model and scaler
lstm_model = load_model(r'c:\Users\joann\Downloads\GreenGauge\lstm_model.h5')
scaler = joblib.load(r'c:\Users\joann\Downloads\GreenGauge\scaler_lstm.joblib')

# Load recent temperature data
recent_data = np.array([
    [4.5], [4.8], [6.4], [8.9], [11.3], [13.2], [14.0], [13.4], [11.8], [9.6],
    # Add additional recent data points up to the 60 needed for lookback
])

# Scale the recent data
recent_data_scaled = scaler.transform(recent_data)

# Reshape for the LSTM model input (1 sample, 60 timesteps, 1 feature)
X_new = np.reshape(recent_data_scaled, (1, recent_data_scaled.shape[0], 1))

# Predict with the LSTM model
lstm_forecast = lstm_model.predict(X_new)

# Inverse scale the prediction to get it back to the original scale
lstm_forecast = scaler.inverse_transform(lstm_forecast)
print("LSTM Forecast:", lstm_forecast)
