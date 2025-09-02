import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation

data = pd.read_csv('../Dataset/Cleandataset/Dataset_car_encoder_v3.csv')

X = data[['brand_encoded','model_encoded','year','engine_size',
          'fuel_type_encoded','transmission_encoded','mileage','body_type_encoded',
          'color_encoded','drive_type_encoded','doors','seats','vehicle_age','tax_status_encoded','has_sunroof',
          'has_gps','service_record','accident_history','mileage_per_year','mileage_log','engine_per_seat',
          'features_count','is_premium_brand','freq_brand','age_x_mileage']]

Y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = LGBMRegressor(
    objective='regression',     
    boosting_type='gbdt',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=256,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    min_data_in_leaf=1000,
    verbose=100,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],   
    eval_metric='rmse',                              
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(100)] 
)

joblib.dump(model, '../Model/Model_car_price_lgbm_v3.pkl')