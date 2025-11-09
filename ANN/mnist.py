import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

#veri setini yükle
(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

print("x_train shape:",x_train.shape,"dtype:",x_train.dtype)
print("label sample",y_train[:10])

print("min/max px",x_train.min(),x_train.max())

counts = np.bincount(y_train)
print("sınıf sayıları ?",counts)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print("min/max normalized",x_train.min(),x_train.max())

x_val, y_val = x_train[-10000:],y_train[-10000:]
x_train, y_train = x_train[-10000:],y_train[-10000:]
print("tra",x_train.shape, "val:",x_val.shape,"test",x_test.shape)

model = keras.Sequential([
    layers.Flatten(input_shape=(28,28), name="flatten"), #28x28 > 784
    layers.Dense(256, activation="relu",kernel_regularizer=regularizers.l2(0.001), name="dense_1_l2"),
    layers.Dropout(0.3, name="dropout_1"),
    layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l2(0.001), name="dense_2_l2"),
    layers.Dropout(0.3, name="dropout_2"),
    layers.Dense(10,activation="softmax",name="output")
])

print("*"*100)
model.summary()
print("*"*100,"katmanlar","*"*100)

for i, layer in enumerate(model.layers):
    print(f"{i} | name = {layer.name} | type {layer.__class__.__name__}")

dense1 = model.get_layer("dense_1_l2")
W, b = dense1.get_weights()
print("dense1 w shape ne ?",W.shape,"bias şekli",b.shape)
print("dense1 ağırlık örneği (ilk 5)",np.round(W.flatten()[:5],4))
print("dense1 bias (ilk5)",np.round(b[:5],4))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy", #etiketleri 0...9 (one-hot değil)
    metrics=["accuracy"]
)

print("optimizer", type(model.optimizer).__name__, "loss",model.loss,"metrics",model.metrics_names)

early = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5,restore_best_weights=True
)
print("-"*50,"eğitim başlıyor","-"*50)
history = model.fit(
    x_train,y_train,
    epochs=30, batch_size=128,
    validation_data=(x_val,y_val),
    callbacks=[early],
    verbose=1
)
print("en iyi val_loss:", np.min(history.history["val_loss"]))

print("-"*50,"test","-"*50)
test_loss, test_acc = model.evaluate(x_test,y_test,verbose=0)
print(f"test loss : {test_loss:.4f} | test acc {test_acc:.4f} ")






























