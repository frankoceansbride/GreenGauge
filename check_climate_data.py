

import pandas as pd

# Load without selecting specific columns to inspect column names
global_data = pd.read_csv(r"c:\Users\joann\Downloads\GreenGauge\climate_change_indicators.csv")
print(global_data.columns)

