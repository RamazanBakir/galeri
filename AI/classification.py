import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import numpy as np
"""
regression -> sayı tahmin eder  -> 2.3 gün
classification -> label tahmin eder -> evet/hayır


ağırlık    boy     tür
4kg        30cm    kedi
20kg       50cm    köpek
1kg        15cm    kedi
100kg      150cm  insan

model bu yapıdan pattern öğreniyor.
10kg 40cm -> büyük ihtimalle köpek sınıfına girer

logistic regression -> evet/hayır -> label üzerine (sayısal değer değildir olasılık üretir) 0.84 


df = pd.DataFrame({
    "hava": ["gunesli","yagmurlu","bulutlu","firtinali","gunesli","yagmurlu"],
    "sicaklik": [30,10,20,12,25,18],
    "ruzgar":[5,20,7,25,6,15],
    "disari": [1,0,1,0,1,0]
})
print(df.head())
X = df[["sicaklik","ruzgar"]]
y = df["disari"]
model = LogisticRegression()
model.fit(X,y)

#yeni gün

df = pd.DataFrame({
    "agirlik":[4,5,20,25],
    "boy": [30,35,50,55],
    "tur": ["kopek","kedi","kopek","kopek"]
})
print(df.head())
X = df[["agirlik","boy"]]
y = df["tur"]

model = KNeighborsClassifier(n_neighbors=3) # K=3
model.fit(X,y)

tahmin = model.predict([[7,40]])
print("tahmin",tahmin)

plt.figure(figsize=(6,6))
plt.scatter(df["agirlik"],df["boy"],c=["green" if t=="kedi" else "blue" for t in df["tur"]], s=100, label="veriler")
#yeni nokta (7,40)
plt.scatter(7,40, c="red",s=150,marker="*",label=f"yeni veri {tahmin}")
plt.show()
print(model.predict([[22,10]]))


decision tree:
max_depth : tree kaç kat derinleşsin
min_samples_split:
min_samples_leaf: 
max_leaf_nodes
ccp_alpha:


df = pd.DataFrame({
    "agirlik":[4,5,20,25],
    "boy": [30,35,50,55],
    "tur": ["kedi","kedi","kopek","kopek"]
})

print(df.head())
X = df[["agirlik","boy"]]
y = df["tur"]

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X,y)

# tahmin + olasılık
yeni = [[7,40]]
print("tahmin", tree.predict(yeni)[0])
print("aaaa",tree.classes_)
print("olasılıklar (sıra:kedi, kopek olabilir)",tree.predict_proba(yeni))

print(export_text(tree,feature_names=list(X.columns)))

plt.figure(figsize=(8,5))
plot_tree(tree,feature_names=list(X.columns), class_names=tree.classes_, filled=True)
plt.show()

print("feature importances: ", dict(zip(X.columns, tree.feature_importances_)))


classification: 
-basit modeller (lnn,logistic regression,decision tree, naive bayes)
-ensemble(random forest, gradient boosting, catboost...)
-gelişmiş(deep learning)(SVM,ANN,cnn/rnn)


df = pd.DataFrame({
    "agirlik":[4,5,20,25],
    "boy": [30,35,50,55],
    "tur": ["kedi","kedi","kopek","kopek"]
})

print(df.head())
X = df[["agirlik","boy"]]
y = df["tur"]

model = RandomForestClassifier(n_estimators=10,oob_score=True,random_state=42)
model.fit(X,y)

#yeni veri
print("tahmin",model.predict([[7,40]])[0])
print("olasılık",model.predict_proba([[7,40]]))
print(f"oob score:", model.oob_score_)

Confusion Matrix
TP (True Positive): 
TN (True Negative):
FP: 
FN:

Sınıflar dengeli  -> Accuracy 
spam filtresi -> precision 
kanser tespiti -> recall 
denge istiyorsak -> f1 score


[1,2,3,4,5,6,7,8,9,10]
ağaç 1 -> [1,2,3,3,4,7,8,8,9,10]
train_test_split()
OOB score 
"""
np.random.seed(42)
agirlik = np.random.randint(4,30,size=100)
boy = np.random.randint(30,70,size=100)
tur = []
for a in agirlik:
    if a < 15:
        tur.append("kedi" if np.random.rand() > 0.1 else "kopek")
    else:
        tur.append("kopek" if np.random.rand() > 0.1 else "kedi")
df = pd.DataFrame({"agirlik":agirlik, "boy":boy,"tur":tur})

print(df.head())
X = df[["agirlik","boy"]]
y = df["tur"]

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.25,random_state=42)
model = RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

etiketler = ["kedi","kopek"]
cm = confusion_matrix(y_test,y_pred,labels=etiketler)

fig,ax = plt.subplots(figsize=(6,5),dpi=120)
im = ax.imshow(cm)

cbar = fig.colorbar(im,ax=ax)
cbar.set_label("adet")

ax.set_title("confusion matrix vs vs")
ax.set_xlabel("tahmin")
ax.set_ylabel("gerçek")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j,i, cm[i,j],
                ha="center",va="center",color="white"
                )
plt.show()
print("gerçek",list(y_test))
print("tahmin",list(y_pred))
print("confusion matrix:", confusion_matrix(y_test,y_pred))
print("acc.",accuracy_score(y_test,y_pred))
print("prec..",precision_score(y_test,y_pred, pos_label="kedi"))
print("rec..",recall_score(y_test,y_pred,pos_label="kedi"))
print("f1..",f1_score(y_test,y_pred,pos_label="kedi"))

"""
acc. 0.88
prec.. 1.0
rec.. 0.7692307692307693
f1.. 0.8695652173913044
renkli confusion matrix heatmap oluşturalım
"""






