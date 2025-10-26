import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib



df = pd.read_csv("data.csv")

print(df.head())

X = df[["yas","boy","kilo", "ders_saati"]]
y = df[["not"]]

numeric_features = ["yas","boy","kilo","ders_saati"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")), #eksik değerleri doldur
    ("scaler",StandardScaler()) #ölçekleme
])

"""#steps zorunlu ad, ...
memory cache klasörü oluşturur (opt)
verbose aşamaları yazdır (opt)"""

preprocess = ColumnTransformer(
    transformers=[
        ("num",numeric_transformer,numeric_features)
    ]
)

model = Pipeline(steps=[
    ("preprocess",preprocess), # önce veriyi işleme
    ("estimator", LinearRegression()) # sonra model
])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
print("ortalama hata değeri",mae)
joblib.dump(model,"ogrenci_listesi.joblib")
