import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers



"""
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])

y = np.array([0,1,1,0]) #XOR örneği

#sequential
model = keras.Sequential([
    layers.Dense(8,activation="relu",input_shape=(2,)),
    layers.Dense(1,activation="sigmoid")
])

#optimizer : adam  |loss
model.compile(
    optimizer = "adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(X, y, epochs=1000, batch_size=4,verbose=0) #validation_split=0.2 buna küçük datada gerek yok

#evaluate
loss, acc = model.evaluate(X,y, verbose=0)
print("kayıp:",loss,"doğruluk:",acc)

y_prob = model.predict(X) #olasılık (0-1)
y_pred = (y_prob >0.5).astype(int)

model = keras.Sequential() #sırayla katman ekleyeceğim.
model.add(layers.Dense(units=2, activation="tanh",input_shape=(2,)))
model.add(layers.Dense(units=1,activation="sigmoid"))

model.summary()

model.compile(
    optimizer = "adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

girdi : 28x28x3 -> 2352
y = w.x +b

model = keras.Sequential([
    layers.Conv2D()
])

Dense(128,activation="relu") -> beyinde 128 nöron var 
Dense(10,activation="softmax") -> 

Conv2D -> kenarları ve desenleri buluyor.
polling -> gereksiz detayları atar
flatten -> tüm resmi düzleştirir
dense -> öğrendiklerinden karar veriyor.

[   [1,2,3,0,1],
    [4,5,6,1,0],
    [7,8,9,2,3],
    [0,1,2,3,4],
    [1,0,1,5,6],
 ]

[   [-1,-1,-1],
    [0,0,0],
    [1,1,1]
 ]
#toplam sonuç = 18
#Padding=same -> girişi ve çıkışı boyut olarak aynı kalır
#Padding=valid -> kenarlarda işlem yapmaz boyut küçülür.


y = 3x+5
"""

tf.random.set_seed(42); np.random.seed(42)

#y = 3x + 5 + noise
x =np.linspace(-2,2,80).reshape(-1,1).astype(np.float32)
y = (3*x + 5 + 0.3*np.random.randn(*x.shape)).astype(np.float32)

model = keras.Sequential([
    keras.Input(shape=(1,)),
    layers.Dense(16,activation="relu"),
    layers.Dense(1)
])


model.compile(optimizer="adam", loss="mse",metrics=["mae"])
model.fit(x,y,epochs=300,batch_size=16,verbose=0)

loss, mae = model.evaluate(x,y,verbose=0)

print("regresyon -> mse: ",loss,"mae:",mae)
print("x=1.5 için tahmin", model.predict(np.array([1.5])) )

model.save("test.keras")

load = keras.models.load_model("benimmodel.keras")



















