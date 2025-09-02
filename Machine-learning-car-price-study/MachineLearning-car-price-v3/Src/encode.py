import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../Dataset/Raw/Dataset_car_v3.csv")

kolom_kategori = ['brand','model','fuel_type','transmission','body_type','color',
                  'drive_type','tax_status']

label_encoders = {}
for kolom in kolom_kategori:
    le = LabelEncoder()
    data[kolom + '_encoded'] = le.fit_transform(data[kolom])
    label_encoders[kolom] = le

joblib.dump(label_encoders, "../Model/label_encoders_car_price_v2.pkl")
data.to_csv("../Dataset/Cleandataset/Dataset_car_encoder_v3.csv", index=False)