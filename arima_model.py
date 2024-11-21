import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Load the dataset
global_data = pd.read_csv(r"c:\Users\joann\Downloads\GreenGauge\climate_change_indicators.csv")

# Filter data by selecting year columns and reshaping them into a time series format
year_columns = [col for col in global_data.columns if col.startswith('F')]
temperature_data = global_data.melt(id_vars=['Country'], value_vars=year_columns,
                                    var_name='Year', value_name='Temperature')

# Convert 'Year' to a datetime format and clean column name for ARIMA
temperature_data['Year'] = temperature_data['Year'].str.extract('(\d+)').astype(int)
temperature_data['Year'] = pd.to_datetime(temperature_data['Year'], format='%Y')
temperature_data.set_index('Year', inplace=True)

# Drop missing values
temperature_data = temperature_data.dropna()

# Ensure monthly frequency if necessary, but given data is yearly, ARIMA will use it as is
temperature_series = temperature_data['Temperature']

# Fit and save ARIMA model
model = ARIMA(temperature_series, order=(5, 1, 0))
model_fit = model.fit()

# Save the model
joblib.dump(model_fit, 'arima_model.joblib')
print("ARIMA model saved as arima_model.joblib")
print("Climate data loaded and processed successfully.")
