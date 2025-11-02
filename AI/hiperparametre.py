from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
hyperparameters

n_estimators -> kaç tane decision tree oluşacak
max_depth -> her tree maks. derinliği -> çok derin olursa overfit, çok az olursa da underfit
n_neighhbors = 1 verirseniz KNN modeli hassas oluyo (overfit)
n_neighhbors = 50 olursa da çok genelleme yapabilir (underfit)
--- hyperparameter tuning ---

GridSearchCV veya RandomizedSearchCv

nasıl yapılır ?
manual search

from sklearn.neighbors import KNeighborsClassifier
for k in [1,3,5,7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,y_train)
    print(k, model.score(X_test,y_test))

param_grid = {
    "n_estimators":[50,100,200],
    "max_depth": [3,5,10],
}

model = RandomForestClassifier(random_state=42)

grid = GridSearchCV(model,param_grid,cv=5)
grid.fit(X_train, y_train)

print("en iyi parametre", grid.best_params_)
print("en iyi skor", grid.best_score_)



X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("eğitim için veri boyutu ", X_train.shape)
print("test için veri boyutu ", X_test.shape)

model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators":[10,50,100],
    "max_depth": [3,5,None],
    "min_samples_split": [2,5]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5, #cross validation kat sayısı !
    n_jobs=-1 #tüm CPU çekirdeklerini kullan
)
grid.fit(X_train,y_train)
print("*"*10,"en iyi parametreler: ",grid.best_params_,"*"*10)
print("*"*10,"en iyi cross-validation skoru: ",grid.best_score_,"*"*10)
print("Test doğruluğu",grid.best_estimator_.score(X_test,y_test))

results = pd.DataFrame(grid.cv_results_)
print(results[["params","mean_test_score"]])
print(results.keys())


model = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": [50,100,150,200,300],
    "max_depth": [3,5,7,10,None],
    "min_samples_split": [2,3,5,10],
    "min_samples_leaf" : [1,2,3,4],
    "max_features": ["sqrt","log2",None]
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
search.fit(X_train,y_train)
print("*"*10,"en iyi parametreler: search ",search.best_params_,"*"*10)
print("*"*10,"en iyi cross-validation skoru: search ",search.best_score_,"*"*10)

best_model = search.best_estimator_
test_acc = accuracy_score(y_test,best_model.predict(X_test))
print("Test doğruluğu", round(test_acc,4))

rf = RandomForestClassifier(random_state=42)


param_dist = {
    "n_estimators":randint(10,200),
    "max_depth": [None,2,4,6,8],
    "min_samples_split": randint(2,10)
}

bias - variance - overfit - underfit

durum             bias         variance
model çok basit   yüksek       düşük   -> underfit
model çok karmaşık düşük       yüksek  **> overfit
model dengeli     düşük       düşül --> ideal (good fit)

"""
#veri => y = sin(x) + noise

np.random.seed(42)
X = np.linspace(0,6,30).reshape(-1,1)
y = np.sin(X).ravel() + np.random.rand(30)*0.2

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

plt.figure(figsize=(10,6))
plt.scatter(X_train,y_train,label="train verisi",color="blue")

for degree in [1,4,15]:
    model = Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("lr",LinearRegression())
    ])
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    plt.plot(X_test,model.predict(X_test), label=f"degree : {degree} | hata = {mse:.2f}")

plt.legend()
plt.title("bias - variance ve overfit/underfit için örnek")
plt.show()








