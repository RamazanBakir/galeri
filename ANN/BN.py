import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt

np.random.seed(42)
N = 20000
X = np.random.randn(N,2).astype("float32")
y = ((X[:,0]*1.5 + X[:,1]*-2.0 + 0.3) >0).astype("float32") # [1.0,0.0,1.0,0.0]

X_train, X_val = X[:1500], X[1500:]
y_train, y_val = y[:1500], y[1500:]

model_plain = keras.Sequential([
    layers.Dense(64, activation="relu",input_shape=(2,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(1,activation="sigmoid")
])

model_plain.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

hist_plain = model_plain.fit(
    X_train,y_train,
    validation_data=(X_val,y_val),
    epochs=30,batch_size=64, verbose=0
)

model_bn = keras.Sequential([
    layers.Dense(64, activation="relu",input_shape=(2,)),
    layers.BatchNormalization(), #fark bu
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(1,activation="sigmoid")
])

model_bn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

hist_bn = model_bn.fit(
    X_train,y_train,
    validation_data=(X_val,y_val),
    epochs=30,batch_size=64, verbose=0
)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(hist_plain.history["val_accuracy"],label="normal")
plt.plot(hist_bn.history["val_accuracy"],label="batchnorm")
plt.xlabel("epoch")
plt.ylabel("val acc.")
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist_plain.history["val_loss"],label="normal")
plt.plot(hist_bn.history["val_loss"],label="batchnorm")
plt.title("loss karşılaştırması")
plt.xlabel("epoch")
plt.ylabel("val loss")
plt.legend()
plt.tight_layout()
plt.show()

















