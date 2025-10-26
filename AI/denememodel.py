import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
"""
a = joblib.load("ogrenci_listesi.joblib")
#print(a.predict([[20,175,68,3]]))

X_new = pd.DataFrame([{
    "yas":20,
    "boy":175,
    "kilo":68,
    "sinif":3,
    "ders_saati": 4
}])
print(a.predict(X_new))

df = pd.DataFrame({
    "saat":[1,2,3,4,5],
    "not": [40,55,70,85,95]
})


X = df[["saat"]]
y = df["not"]

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)
print(X_poly)

model = LinearRegression()
model.fit(X_poly,y)
print(model.predict(poly.transform([[3]])))
print(model.predict(poly.transform([[4]])))

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("reg",LinearRegression())
])

model.fit(X,y)
print(model.predict([[4]]))
"""

df = pd.DataFrame({
    "saat":[1,2,3,4],
    "not": [50,55,60,85]
})

X = df[["saat"]].values
y = df["not"].values

x_grid = np.linspace(X.min()-0.5,X.max()+0.5, 200).reshape(-1,1)

#linear regresyon
lin_model = LinearRegression()
lin_model.fit(X,y)
y_lin = lin_model.predict(x_grid)

plt.figure(figsize=(6,4))
plt.scatter(X,y,label="veri")
plt.plot(x_grid,y_lin,label="Linear")
plt.title("Linear reg.")
plt.xlabel("Saat")
plt.ylabel("not")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#polynomal reg.

poly_model = Pipeline([
    ("poly",PolynomialFeatures(degree=2)),
    ("lin",LinearRegression())
])

poly_model.fit(X,y)
y_poly = poly_model.predict(x_grid)

plt.figure(figsize=(6,4))
plt.scatter(X,y,label="veri")
plt.plot(x_grid,y_poly,label="poly. degre2",color="red")
plt.title("poly reg.")
plt.xlabel("Saat")
plt.ylabel("not")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


svr_model = Pipeline([
    ("scaler",StandardScaler()),
    ("svr",SVR(kernel="rbf", C=10, epsilon=2))
])
svr_model.fit(X,y)
y_svr = svr_model.predict(x_grid)

plt.figure(figsize=(6,4))
plt.scatter(X,y,label="veri")
plt.plot(x_grid,y_svr,label="svr reg.(rbf)",color="green")
plt.title("svr reg.")
plt.xlabel("Saat")
plt.ylabel("not")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


print(svr_model.predict([[3]]))






