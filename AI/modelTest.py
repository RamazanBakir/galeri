import joblib
import pandas as pd

model = joblib.load("kargo_teslim.joblib")

yeni = pd.DataFrame([{
    "mesafe_km": 1131,
    "paket_agirligi": 8.0,
    "kargo_tipi": "standart",
    "hava_durumu": "yagmurlu"
}])

print("yeni tahmin ", model.predict(yeni)[0])
# 1131             8.0   standart    yagmurlu               4.20