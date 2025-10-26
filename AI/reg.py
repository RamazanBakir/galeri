import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
"""
#model seç
model = LinearRegression()

#öğrenme verisi
X = [[50],[70],[100]] #m2 değerleri
y = [1000, 1500, 2100] #kira fiyatı

#öğret:
model.fit(X,y)

#yeni bir evin kira fiyatını tahmin etsin.
print(model.predict([[80]])[0])
print(model.predict([[40]])[0])

#veri (m2, kira)
X = np.array([[50],[70],[100],[120]])
y = np.array([1000,1500,2100,2400])

model = LinearRegression()
model.fit(X,y)

plt.figure(figsize=(7,4))
plt.scatter(X,y,label="veri noktaları")

x_line = np.linspace(X.min(),X.max(),100).reshape(-1,1)
y_line = model.predict(x_line)
plt.plot(x_line,y_line,label="Öğrenilen Çizgi - Linear regresyon")


x_new = np.array([[80]])
y_new = model.predict(x_new)[0]
plt.scatter([x_new[0,0]], [y_new],marker="x",s=100,label=f"80m2 tahmini fiyatı {y_new}")


plt.title("kira tahmini")
plt.xlabel("m2")
plt.ylabel("kira")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("linear_sonuc.pdf",dpi=150)
plt.show()

#öğrenme verisi
X = np.array([[50],[60],[70],[80],[90],[100]]) #m2 değerleri
y = np.array([950,1000,1500,2100,2400,2500]) #kira fiyatı

#eğitim ve test böl
X_tr, X_te,y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression().fit(X_tr,y_tr)

y_pred = model.predict(X_te)

plt.figure(figsize=(6,6))
plt.scatter(y_te, y_pred,label="test noktaları") #x : gerçek y:tahmin

min_v = min(y_te.min(),y_pred.min())
max_v = max(y_te.max(),y_pred.max())
plt.plot([min_v, max_v], [min_v,max_v],"--",label="diyagonal (y=x)")


plt.title("gerçek vs tahmin (test seti)")
plt.xlabel("gerçek")
plt.ylabel("tahmin")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
r2 skoru -> 1'e ne kadar yakınsa o kadar iyi'
hata miktarı -> 0'a ne kadar yakınsa o kadar iyi'
r2 = r2_score(y_te,y_pred)
mae = mean_absolute_error(y_te,y_pred) #ortalama mutlak hata
mse = mean_squared_error(y_te,y_pred) #ortalama kare hata
sq = np.sqrt(mse) #kareköklü hata
print("r2 skoru ? : ",r2,"mae",mae,"mse",mse,"sq",sq)
"""

df = pd.read_csv("test.csv")
print(df.head())














