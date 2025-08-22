import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

model = joblib.load("Result/model_machine_learning_linear_regression_car_price_v1.pkl")

data = pd.read_csv("Result/Dataset_car_encoded.csv")

X = data[['brand_encoded', 'model_encoded', 'year', 'engine_size',
          'fuel_type_encoded', 'transmission_encoded', 'mileage',
          'body_type_encoded', 'color_encoded', 'drive_type_encoded',
          'doors', 'seats', 'vehicle_age', 'tax_status_encoded', 'has_sunroof_encoded',
          'has_gps_encoded', 'service_record_encoded', 'accident_history_encoded']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

prediksi = model.predict(X_test)

mse = mean_squared_error(y_test, prediksi)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, prediksi)

print(f'Prediksi: {prediksi[:5]}')  
print(f'Real: {y_test.values[:5]}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R² Score: {r2:.2f}')


