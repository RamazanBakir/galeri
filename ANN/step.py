import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""
#1-) veri : "toplamı >=1 is 1, değilse 0"
rng = np.random.default_rng(42)

X = rng.random((1000,2)).astype("float32")
y = ((X[:,0] + X[:,1]) >= 1.0).astype("float32").reshape(-1,1)

def make_model():
    model = keras.Sequential([
        layers.Dense(8, activation="relu",input_shape=(2,)),
        layers.Dense(1,activation="sigmoid")
    ])
    return model

#deneme a : adam + binary_crossentropy (doğru kombinasyon)

model_a = make_model()
model_a.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
hist_a = model_a.fit(X,y,epochs=40,batch_size=32,validation_split=0.2,verbose=0)
loss_a, acc_a = model_a.evaluate(X,y,verbose=0)

#deneme b : SGD + mse(sınıflandırma için uygun değil bilerek gösterelim)
model_b = make_model()
model_b.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),loss="mse",metrics=["accuracy"])
hist_b= model_b.fit(X,y,epochs=40,batch_size=32,validation_split=0.2,verbose=0)
loss_b, acc_b = model_b.evaluate(X,y,verbose=0)

print(f"deneme a > acc: {acc_a:.3f}, loss: {loss_a:.4f}")
print(f"deneme b > acc: {acc_b:.3f}, loss: {loss_b:.4f}")

"""
rng = np.random.default_rng(42)

X = rng.random((2000,2)).astype("float32")
y = ((X[:,0] + X[:,1]) >= 1.0).astype("float32").reshape(-1,1)

X_train, y_train = X[:120],y[:120]
X_val, y_val = X[1600:],y[1600:]

model_under = keras.Sequential([
    layers.Dense(2,activation="relu",input_shape=(2,)),
    layers.Dense(1,activation="sigmoid")
])
model_under.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
hist_under = model_under.fit(
    X_train, y_train,
    validation_data=(X_val,y_val),
    epochs=20,batch_size=32,verbose=0
)

print("underfit : ",
      model_under.evaluate(X_train,y_train,verbose=0)[1],
      model_under.evaluate(X_val,y_val,verbose=0)[1]
      )

model_over = keras.Sequential([
    layers.Dense(128,activation="relu",input_shape=(2,)),
    layers.Dense(128,activation="relu"),
    layers.Dense(64,activation="relu"),
    layers.Dense(1,activation="sigmoid"),
])

model_over.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

hist_over = model_over.fit(
    X_train, y_train,
    validation_data=(X_val,y_val),
    epochs=400,batch_size=16,verbose=0
)

print("overfit : ",
      model_over.evaluate(X_train,y_train,verbose=0)[1],
      model_over.evaluate(X_val,y_val,verbose=0)[1]
      )


# Dropout, L2, EarlyStopping
layers.Dropout(0.3)
layer.Dense(64,activation="relu",kernel_regularizer=regularizers.l2(0.001))


early= keras.callbacks.EarlyStopping(
    patience=10 #10 epochs iyileşme yoksa dur.
    restore_best_weights = True #en iyi ağırlıklara dön
)















