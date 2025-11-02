import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import FunctionTransformer,StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.pipeline import Pipeline
"""
cross validation

"""
"""df = pd.DataFrame(data.data,columns=data.feature_names)
df["target"] = data.target
print(df.head())
data = fetch_california_housing()
X = data.data
y = data.target

model = LinearRegression()
scores = cross_val_score(model,X,y,cv=5)

print("her katın skoru ? (R2)", scores)
print("ortalama doğruluk ? (R2)", np.mean(scores))

#dropout
train_acc = model.score(X_train, y_train)   #0.99
test_acc = model.score(X_test, y_test)    #0.82

data = load_iris(as_frame=True)
print(data)
print("*"*100)
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
print(df.head())


X,y = load_iris(return_X_y=True)
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)

for depth in [1,2,3,5,10,20]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train,y_train)
    train_acc = tree.score(X_train,y_train)
    test_acc = tree.score(X_test, y_test)
    print(f"depth {depth}: train: {train_acc} test: {test_acc}")

"""
df = pd.DataFrame({
    "fiyat": [10,20,15,50,60,45,12,18],
    "adet": [1,2,1,5,1,2,4,3],
    "sehir": ["izmir","istanbul","ankara","izmir","ankara","istanbul","izmir","ankara"],
    "tarih" : pd.to_datetime([
        "2025-01-02","2025-02-10","2025-02-15","2025-03-01",
        "2025-04-11","2025-04-12","2025-05-01","2025-05-22",
    ]),
    "satis": [0,1,0,1,1,1,0,1] #hedef : satış başarılı mı ? e/h
})

print(df.head())

X_base = df[["fiyat","adet"]]
y = df["satis"]

X_tr, X_te,y_tr,y_te = train_test_split(X_base,y, test_size=0.4,random_state=42)

base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_tr,y_tr)
y_pred_base = base_model.predict(X_te)
print("-"*10,"özellik eklemeden ham veriyle","-"*10)
print(classification_report(y_te,y_pred_base,digits=3))

"tutar = fiyat * adet"
"ay = tarih.month"
"sehir = One-hot (yazıyı sayıya çevir)"

df_fe = df.copy()
df_fe["tutar"] = df_fe["fiyat"] * df_fe["adet"]
df_fe["ay"] = df_fe["tarih"].dt.month

df_fe = pd.get_dummies(df_fe, columns=["sehir"], drop_first=True)
print(df_fe.head())

use_cols = ["fiyat","adet","tutar","ay","sehir_istanbul","sehir_izmir"]
X_fe = df_fe[use_cols]
y = df_fe["satis"]

X_tr, X_te,y_tr,y_te = train_test_split(X_fe,y, test_size=0.4,random_state=42)

fe_model = LogisticRegression(max_iter=1000)
fe_model.fit(X_tr,y_tr)
y_pred_fe = fe_model.predict(X_te)

print("-"*10,"f_e eklendikten sonra veriyle","-"*10)
print(classification_report(y_te,y_pred_fe,digits=3))



def add_features(X: pd.DataFrame) -> pd.DataFrame:
    out= X.copy()
    out["tutar"] = out["fiyat"] * out["adet"]
    out["ay"] = out["tarih"].dt.month
    out = out.drop(columns=["tarih"])
    return out

feat_gen = FunctionTransformer(add_features)
num_cols = ["fiyat","adet","tutar","ay"]
cat_col = ["sehir"]
pre = ColumnTransformer([
    ("num",StandardScaler(),num_cols),
    ("cat",OneHotEncoder(drop="first", handle_unknown="ignore"), cat_col)
])

pipe = Pipeline([
    ("feat",feat_gen), #yeni sütunlar eklensin
    ("prep",pre), #ölçeklendir + one-hot
    ("model",LogisticRegression(max_iter=1000))
])

X = df[["fiyat","adet","sehir","tarih"]]
y = df["satis"]

X_tr, X_te,y_tr,y_te = train_test_split(X,y, test_size=0.4,random_state=42, stratify=y)
pipe.fit(X_tr,y_tr)
print("pipeline eklendikten sonra acc.", pipe.score(X_te,y_te))























