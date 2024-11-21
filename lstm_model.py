import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the dataset
global_data = pd.read_csv(r"c:\Users\joann\Downloads\GreenGauge\climate_change_indicators.csv")

# Reshape year columns into a time series format
year_columns = [col for col in global_data.columns if col.startswith('F')]
temperature_data = global_data.melt(id_vars=['Country'], value_vars=year_columns,
                                    var_name='Year', value_name='Temperature')

# Convert 'Year' to datetime format and set as index
temperature_data['Year'] = temperature_data['Year'].str.extract('(\d+)').astype(int)
temperature_data['Year'] = pd.to_datetime(temperature_data['Year'], format='%Y')
temperature_data.set_index('Year', inplace=True)
temperature_data = temperature_data.dropna()

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(temperature_data[['Temperature']])

# Prepare data with a 60-year lookback period
def prepare_lstm_data(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = prepare_lstm_data(scaled_data, lookback=60)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM

# Build and compile the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)
model.save(r'c:\Users\joann\Downloads\GreenGauge\lstm_model.h5')
joblib.dump(scaler, r'c:\Users\joann\Downloads\GreenGauge\scaler_lstm.joblib')
print("LSTM model and scaler saved successfully.")
