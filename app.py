from flask import Flask, render_template, request, redirect,  url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__) 
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Ensure the necessary directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('./models', exist_ok=True)

scaler = MinMaxScaler(feature_range=(0, 1))

def prepare_data(file_path):
    data = pd.read_csv(file_path)
    year_columns = [col for col in data.columns if col.startswith('F')]
    temperature_data = data.melt(id_vars=['Country'], value_vars=year_columns, 
                                 var_name='Year', value_name='Temperature')
    temperature_data['Year'] = temperature_data['Year'].str.extract(r'(\d+)').astype(int)  # Updated with raw string
    temperature_data['Year'] = pd.to_datetime(temperature_data['Year'], format='%Y')
    temperature_data.set_index('Year', inplace=True)
    return temperature_data[['Temperature']].dropna()

def train_arima_model(data):
    temperature_series = data['Temperature']
    model = ARIMA(temperature_series, order=(5, 1, 0))
    model_fit = model.fit()
    joblib.dump(model_fit, './models/arima_model.joblib')

def train_lstm_model(data):
    global scaler
    scaled_data = scaler.fit_transform(data[['Temperature']])
    X, y = [], []
    lookback = 60
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)
    model.save('./models/lstm_model.h5')
    joblib.dump(scaler, './models/scaler_lstm.joblib')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], "climate_data.csv")  # Save as 'climate_data.csv'
            file.save(file_path)
            data = prepare_data(file_path)
            plt.figure()
            data.plot()
            plt.title('Uploaded Temperature Data')
            plt.xlabel('Year')
            plt.ylabel('Temperature')
            plt.savefig('./static/uploads/historical_data.png')
            return render_template("index.html", data_uploaded=True)
    return render_template("index.html", data_uploaded=False)

@app.route("/train", methods=["POST"])
def train():
    model_type = request.form['model']
    data = prepare_data('./static/uploads/climate_data.csv')  
    if model_type == "ARIMA":
        train_arima_model(data)
        model_info = "ARIMA model trained and saved."
    elif model_type == "LSTM":
        train_lstm_model(data)
        model_info = "LSTM model trained and saved."
    return render_template("result.html", model_info=model_info)

@app.route("/forecast", methods=["POST"])
def forecast():
    model_type = request.form['model']
    recent_data = prepare_data('./static/uploads/climate_data.csv')['Temperature'][-60:].values.reshape(-1, 1)
    forecast_result = None

    if model_type == "ARIMA":
        model = joblib.load('./models/arima_model.joblib')
        forecast_result = model.forecast(steps=10)
    elif model_type == "LSTM":
        model = load_model('./models/lstm_model.h5')
        scaler = joblib.load('./models/scaler_lstm.joblib')
        recent_data_scaled = scaler.transform(recent_data)
        X_recent = np.reshape(recent_data_scaled, (1, 60, 1))
        forecast_result = model.predict(X_recent)
        forecast_result = scaler.inverse_transform(forecast_result)
    
    plt.figure()
    plt.plot(forecast_result, label="Forecast")
    plt.title(f"{model_type} Forecast")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend()
    plt.savefig('./static/uploads/forecast_result.png')
    return render_template("result.html", forecast=True)

if __name__ == "__main__":
    app.run(debug=True)
