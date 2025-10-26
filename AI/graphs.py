import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
"""
#nokta grafiği (scatter)
metrekare = [60,80,90,120,150]
fiyat = [1.2,1.3,1.5,2.4,3.1]

plt.scatter(metrekare,fiyat)
plt.title("burası başlık alanı")
plt.xlabel("m2")
plt.ylabel("fiyat")
plt.grid(True)
plt.show()

#bar chart
random.seed(42)
notlar = [random.randint(0,100) for _ in range(100)]

plt.hist(notlar, bins=10)
plt.title("burası başlık alanı")
plt.xlabel("not")
plt.ylabel("kişi sayısı")
plt.grid(axis="y", linestyle=":")
plt.show()

gun = [1,2,3,4,5,6,7]
a = [10,12,34,12,56,23,11]
b = [50,22,12,53,21,12,41]

plt.plot(gun,a,label="Mağaza A")
plt.plot(gun,b,label="Mağaza B")
plt.title("haftalık satışı vs.")
plt.xlabel("gün")
plt.ylabel("satış")
plt.legend() #labelleri göster
plt.grid(True)
plt.savefig("satis_grafik.png",dpi=150, bbox_inches="tight")
plt.show()

b = [1,2,3,4,5]
a = np.array([1,2,3,4,5])
print(a * 2)
print(a.mean())

ss = pd.read_csv("veriler.csv")
print(ss)
ss.to_json("ramazan.json", index=False)
"""
df = pd.read_csv("sales.csv", parse_dates=["tarih"])
df["tutar"] = df["adet"] * df["fiyat"]


"""
df["adet"] = df["adet"].fillna(0)
df = df[df["fiyat"] >= 0]
"""
df["adet"] = df["adet"].apply(lambda x: 0 if pd.isna(x) else x)
# fiyat boşsa ortalama ile doldursun, negatifse pozitif yapsın.
ort_fiyat = df["fiyat"].mean()
df["fiyat"] = df["fiyat"].apply(lambda x: ort_fiyat if pd.isna(x) else abs(x))
df["tutar"] = df.apply(lambda row: row["adet"] * row["fiyat"], axis=1)
ozet = df.groupby("ürün", as_index=False)["tutar"].sum().sort_values("tutar",ascending=False)
print(df)
print(ozet)

plt.figure(figsize=(6,4), dpi=120)
plt.bar(ozet["ürün"], ozet["tutar"])
plt.title("ürün bazlı toplam ciro")
plt.xlabel("ürün")
plt.ylabel("toplam tutar")
plt.tight_layout()


df_ts = df.set_index("tarih").sort_index()
aylik = df_ts["tutar"].resample("M").sum() #ay ay toplam ciro
print(aylik)

plt.figure(figsize=(6,4),dpi=120)
plt.plot(aylik.index, aylik.values, marker="o")
plt.title("aylık toplam ciro")
plt.tight_layout()
plt.show()

top3 = ozet.head(3)
print(top3)

plt.figure(figsize=(6,4),dpi=120)
plt.bar(top3["ürün"], top3["tutar"])
plt.title("en çok ciro yapan 3 ürün")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

"""
# tutar = adet * fiyat

X -> input (adet ve fiyat) | features/özellik
y -> tahmin edilmek istenen değerim (tutar)  | target/hedef  

sklearn ne yapar ?
veriyi öğretir
tahmin yapar
sınıflandırma yapar (spam/değil)
gruplama yapar (müşteri segmenti gibi)
temel veri dönüştürme (standardizasyon)

# her model hep aynı 3 adımı kullanır 
modeli seç -> model = Model()
öğret -> model.fit(X,y)
tahmin yap -> model.predict(....)

Linear Regression (doğrusal regresyon)

fiyat = a * m2 + b
"""

