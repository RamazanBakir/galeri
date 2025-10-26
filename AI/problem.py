import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
"""
df = pd.read_csv("saat.csv")
print(df.head())

X = df[["saat"]]
y = df[["not"] ]#tahmin edilecek sütun

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

#test verisi üzerinde tahmin
y_pred = model.predict(X_test)

#model değerlendirmesi
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("ortamala hata ", mae)
print("r2 başarı",r2)

joblib.dump(model,"a.joblib")


#model = joblib.load("model.joblib") #alabiliriz. ?
model = joblib.load("a.joblib")
tahmin = model.predict(pd.DataFrame({"saat":[5]}))
#print(model.predict([[5]]))
print(tahmin)

a = model.coef_[0]
b = model.intercept_
print("modelin formülü",a)
print("modelin formülü",b)
"""























