from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
"""
mesafe(km)
paket_agirligi(kg)
kargo_tipi(standart,hizli,ekspres)
hava_durumu(0=iyi, 1=yağmurlu, 2=fırtına)
teslim_suresi(gun)

OneHotEncoder

renk     ağırlık  etiket   kod
kırmızı   10      elma    0
yeşil     15      muz    1
mavi      12     kivi    2
label encoding
one hot encoding


kırmızı mavi yeşil
1       0     0
0       1    0
0       0   1

df = pd.DataFrame({
    "renk": ["kırmızı","yeşil","mavi"]
})

#OneHotEncoder nesnesi oluşturuyoruz
encoder = OneHotEncoder()

encoded = encoder.fit_transform(df[["renk"]])

df_encoded = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(["renk"]))
print(df_encoded)

Girdi(X) : mesafe, km.....
Çıktı(y): 2 günde teslim edilir vb.
"""

np.random.seed(42)

N = 500

mesafe_km = np.random.randint(5,1500,size=N)
paket_agirligi = np.round(np.random.uniform(0.1,30, size=N))

kargo_tipleri = np.random.choice(["standart","hizli","ekspress"], size=N, p=[0.55,0.3,0.15])
hava_durumu = np.random.choice(["iyi","yagmurlu","firtinali"],size=N, p=[0.7,0.25,0.05])

df = pd.DataFrame({
    "mesafe_km":mesafe_km,
    "paket_agirligi":paket_agirligi,
    "kargo_tipi":kargo_tipleri,
    "hava_durumu":hava_durumu
})

print(df.head())
print("-"*1000)
tip_katsayi = df["kargo_tipi"].map({"standart":1.2,"hizli":0.9,"ekspress":0.7})
hava_katsayi = df["hava_durumu"].map({"iyi":0.0,"yagmurlu":0.4,"firtinali":1.0})
#loc ve iloc
gurultu = np.random.normal(loc=0.0, scale=0.2, size=N)

teslim_suresi =(df["mesafe_km"]/500 +(df["paket_agirligi"]/20 + tip_katsayi + hava_katsayi + gurultu))
teslim_suresi = np.clip(teslim_suresi,0.3,None)

df["teslim_suresi_gun"] = np.round(teslim_suresi,2)
print(df.head())

print(df.shape) #boyut
print(df.dtypes) #veri tipleri
print(df.isna().sum()) #eksik var mı ?
print(df.describe()) #sayısal özet
print(df["kargo_tipi"].value_counts())
print(df["hava_durumu"].value_counts())


df_dirty = df.copy()
na_idx = np.random.choice(df_dirty.index, size=20, replace=False)
df_dirty.loc[na_idx,"paket_agirligi"] = np.nan

neg_idx = np.random.choice(df_dirty.index,size=10,replace=False)
df_dirty.loc[neg_idx,"mesafe_km"] = -np.abs(df_dirty.loc[neg_idx,"mesafe_km"])

na_cat_idx = np.random.choice(df_dirty.index,size=10,replace=False)
df_dirty.loc[na_cat_idx,"hava_durumu"] = np.nan
print("-"*100)
print(df_dirty.isna().sum())
print((df_dirty["mesafe_km"] < 0).sum(),"şu kadar negatif mesafe...")

df_clean = df_dirty.copy()
df_clean["mesafe_km"] = df_clean["mesafe_km"].abs()

mod_hava = df_clean["hava_durumu"].mode().iloc[0]
df_clean["hava_durumu"] = df_clean["hava_durumu"].fillna(mod_hava)

medyan_agirlik = df_clean["paket_agirligi"].median()
df_clean["paket_agirligi"] = df_clean["paket_agirligi"].fillna(medyan_agirlik)

print(df_clean.isna().sum())
print((df_clean["mesafe_km"] < 0).sum(),"şu kadar negatif mesafe...")
print(df_clean.head())
print("-"*100)
#kargo tipi değişse ya da hava durumu değişse bu nasıl etkiler ?
print(df_clean.groupby("kargo_tipi")["teslim_suresi_gun"].mean().sort_values())
print(df_clean.groupby("hava_durumu")["teslim_suresi_gun"].mean().sort_values())

X = df_clean[["mesafe_km","paket_agirligi","kargo_tipi","hava_durumu"]]
y = df_clean["teslim_suresi_gun"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


num_cols = ["mesafe_km","paket_agirligi"]
cat_cols = ["kargo_tipi","hava_durumu"]

preprocess = ColumnTransformer([
    ("num",StandardScaler(), num_cols),
    ("cat",OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

linreg_model = Pipeline([
    ("prep", preprocess),
    ("model",LinearRegression())
])

linreg_model.fit(X_train,y_train)


"""
y_pred = linreg_model.predict(X_test)
print(f"gerçek değerlerim {y_test.iloc[:10].round(2)}:",y_test.head(10).to_list())
print("-"*100)
print(f"tahminler {y_pred}", np.round(y_pred[:10],2))
"""

y_pred = linreg_model.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print(f"MAE: {mae:.3f} gün")
print(f"mse: {mse:.3f} gün")
print(f"rmse: {rmse:.3f} gün")
print(f"r2: {r2:.3f} gün")

"""
plt.figure(figsize=(5,5))
plt.scatter(y_test,y_pred)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([mn,mx], [mn,mx], linestyle="--")
plt.xlabel("gerçek")
plt.ylabel("tahmin")
plt.tight_layout()
plt.show()
"""

rf_model = Pipeline([
    ("prep",preprocess),
    ("model", RandomForestRegressor(
        n_estimators=500, random_state=42, n_jobs=-1
    ))
])
rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"r2 -> ", r2_score(y_test,y_pred_rf))
print(f"mae -> ", mean_absolute_error(y_test,y_pred_rf))

#en iyi bulduğunuz modeli kaydedin...
joblib.dump(linreg_model,"kargo_teslim.joblib")

