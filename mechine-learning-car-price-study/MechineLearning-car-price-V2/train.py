import pandas as pd 
import joblib
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

os.makedirs('Result', exist_ok=True)

data = pd.read_csv('Dataset/Dataset_car_v2.csv')

kolom_kategori = ['brand','model','fuel_type','transmission','body_type','color',
                  'drive_type','tax_status']

label_encoders = {}
for kolom in kolom_kategori:
    le = LabelEncoder ()
    data[kolom + '_encoded'] = le.fit_transform(data[kolom])
    label_encoders[kolom] = le

joblib.dump(label_encoders, 'Result/label_encoders_car_price_v2.pkl')
data.to_csv('Result/Dataset_car_encoder_v2.csv', index = False)    

X = data[['brand_encoded','model_encoded','fuel_type_encoded','transmission_encoded','body_type_encoded',
          'color_encoded','drive_type_encoded','tax_status_encoded']]

Y = data[['year','engine_size','mileage','doors','seats','vehicle_age','has_sunroof','has_gps','service_record',
         'accident_history','price','mileage_per_year','mileage_log','engine_per_seat','features_count','is_premium_brand',
         'freq_brand','age_x_mileage']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


model = MultiOutputRegressor(XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    tree_method='hist',
    eval_metric='rmse'
))

model.fit(X_train, y_train)

joblib.dump(model, 'Result/MechineLearning-car-price-v2-XGBRegressor.pkl')