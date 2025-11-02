import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
"""
X, y = datasets.make_blobs(n_samples=30,centers=2,random_state=6,cluster_std=1.0)

model = SVC(kernel="linear") #linear -> düz çizgi ike ayır | poly, rbf, sigmoid
model.fit(X,y)

plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=y, s=60)

plt.scatter(model.support_vectors_[:,0],
            model.support_vectors_[:,1],
            s=120,label="destek vektörler.")

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0],xlim[1],30)
yy = np.linspace(ylim[0],ylim[1],30)

YY, XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX,YY,Z,colors="k",levels=[-1,0,1], alpha=0.8,
           linestyles=["--","-","--"]
           )

plt.legend()
plt.title("svm ile class... sınırı")
plt.show()

2 boyutlu (x,y) 
mavi noktalar = 0 -> a
kırmızı noktalar = 1 -> b

"""

X = np.array([
    [1,2],[2,3],[2,1],[3,2], #grup a (0)
    [6,7], [7,8],[8,6],[7,6] #grup b (1)
])
y = np.array([0,0,0,0,1,1,1,1])

model = SVC(kernel="linear") #düz çizgi!!!
model.fit(X,y)

plt.figure(figsize=(6,5))
plt.scatter(X[:,0],X[:,1], c=y, cmap="coolwarm", s=80)

plt.scatter(model.support_vectors_[:,0],
            model.support_vectors_[:,1],
            s=150, facecolors="none",edgecolors="black",label="destek vektörleri")

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0],xlim[1],30)
yy = np.linspace(ylim[0],ylim[1],30)
YY, XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX,YY,Z,colors="k",levels=[-1,0,1],linestyles=["--","-","--"])

plt.legend()
plt.title("svm örnek")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()






