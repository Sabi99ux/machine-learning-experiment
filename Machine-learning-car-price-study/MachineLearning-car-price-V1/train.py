import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

os.makedirs("Result", exist_ok=True)

data = pd.read_csv('MechineLearning-car-price-V1/Dataset/Dataset_car.csv')

kolom_kategori = ['brand', 'model', 'fuel_type', 'transmission', 'body_type', 'color', 'drive_type','tax_status',
                 'has_sunroof','has_gps','service_record','accident_history' ]

label_encoders = {}
for kolom in kolom_kategori:
    le = LabelEncoder()
    data[kolom + '_encoded'] = le.fit_transform(data[kolom])
    label_encoders[kolom] = le

joblib.dump(label_encoders, 'Result/label_encoders_car_price.pkl')

data.to_csv('Result/Dataset_car_encoded.csv', index=False)    

X = data[['brand_encoded', 'model_encoded', 'year', 'engine_size',
          'fuel_type_encoded', 'transmission_encoded', 'mileage',
          'body_type_encoded', 'color_encoded', 'drive_type_encoded',
          'doors', 'seats','vehicle_age','tax_status_encoded','has_sunroof_encoded',
          'has_gps_encoded','service_record_encoded','accident_history_encoded']]
y = data['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
prediksi = model.predict(X_test)

joblib.dump(model, 'Result/model_machine_learning_linear_regression_car_price_v1.pkl')

