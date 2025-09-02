import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

model = joblib.load('../Model/Model_car_price_lgbm_v3.pkl')

data = pd.read_csv('../Dataset/Cleandataset/Dataset_car_encoder_v3.csv')

X = data[['brand_encoded','model_encoded','year','engine_size',
          'fuel_type_encoded','transmission_encoded','mileage','body_type_encoded',
          'color_encoded','drive_type_encoded','doors','seats','vehicle_age','tax_status_encoded','has_sunroof',
          'has_gps','service_record','accident_history','mileage_per_year','mileage_log','engine_per_seat',
          'features_count','is_premium_brand','freq_brand','age_x_mileage']]

Y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

prediksi = model.predict(X_test)

mse = mean_squared_error(y_test, prediksi)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, prediksi)

for i in range(5):
    print(f"Prediksi: {prediksi[i]:.2f}")
    print(f"Real    : {y_test.values[i]:.2f}")

print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

hasil_evaluasi = {
    "RMSE": float(rmse),
    "R2": float(r2),
    "contoh_prediksi": [
        {"prediksi": float(prediksi[i]), "real": float(y_test.values[i])}
        for i in range(5)
    ]
}

with open("../Result/evaluasi_lgbm_v3.json", "w") as f:
    json.dump(hasil_evaluasi, f, indent=4)