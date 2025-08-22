import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

model = joblib.load('Result/MechineLearning-car-price-v2.pkl')

data = pd.read_csv('Result/Dataset_car_encoder_v2.csv')

X = data[['brand_encoded','model_encoded','fuel_type_encoded','transmission_encoded','body_type_encoded',
          'color_encoded','drive_type_encoded','tax_status_encoded']]

Y = data[['year','engine_size','mileage','doors','seats','vehicle_age','has_sunroof','has_gps','service_record',
         'accident_history','price','mileage_per_year','mileage_log','engine_per_seat','features_count','is_premium_brand',
         'freq_brand','age_x_mileage']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

prediksi = model.predict(X_test)

mse = mean_squared_error(y_test, prediksi, multioutput='raw_values')
rmse = np.sqrt(mse)
r2 = r2_score(y_test, prediksi, multioutput='raw_values')

r2_total = r2_score(y_test, prediksi, multioutput='variance_weighted')
rmse_total = np.sqrt(mean_squared_error(y_test, prediksi))

for i in range(5):
    print(f"Prediksi: {prediksi[i]}")
    print(f"Real   : {y_test.values[i]}")

print("\nEvaluasi per target:")
for i, col in enumerate(Y.columns):
    print(f"{col}: RMSE = {rmse[i]:.2f}, R² = {r2[i]:.2f}")


print(f"R² Total: {r2_total:.4f}")
print(f"RMSE Total: {rmse_total:.2f}")