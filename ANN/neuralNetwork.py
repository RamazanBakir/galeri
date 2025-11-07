"""
hava sıcaklığı      sıcak mı soğuk mu ?    30C
yağmur var mı ?     evet/hayır             0(hayır)
rüzgar              şiddetli mi ?           1 (evet)   output : 1(çık)

1 tane input
dışarı çık (1) ya da evde kal (0)

sıcaklık : önemli (+0.6)
yağmur : çok önemli (-0.9)
rüzgar : orta (-0.3)

(30 x 0.6) + (0 x -0.9) + (1 x -0.3) = 17.7
0 ile 1 arasında karar ver :(sigmoid).None
sonuç eğer 0.95 çıkarsa : dışarı çık (1)
sonuç 0.1 çıkarsa : evde kal (0)

ağırlık vardır : weight
weight üzerine ayarlamalar yapabiliriz.

backpropagation

yeni ağırlık = eski ağırlık - öğrenme oranı x türev (hata)
hata yüksekse = ağırlık daha fazla değişir
hata düşükse = küçük düzeltme yapılır


output = ağırlıklı girişlerin toplamı
--- activation function ---
sigmoid : 0-1 arası -> çıktı katmanında, olasılık hesaplamalarında
tanh : -1 - 1 arası -> hidden layer
ReLU : max(0,x) -> hızlı öğrenme -> (örn. 3,-2,0.5) = f(x) = max(0,x) -> sonuç kendisi olur | eğer negatifse 0
Leaky ReLU

-5 : 0  -0.05
-1 : 0   -0.01
0 :  0    0
2 :  2   2

nöron çalışma mantığı
her nöron, input (x) ağırlık(w) çarpar, toplar, üzerine bias(b) ekler
sonrasında aktivasyon fonksiyonu uygular.


forwardpropagation


epoch : tüm veriyi baştan sona gezmek demek (1tur)
batch : veriyi küçğk parçalarabölüp her parçada güncelleme yapmak
Learning Rate : ağırlıkları ne kadar "büyük anlamda" güncellememiz lazım ?
train/validation/test :
metrics : accuracy, f1,mae,mse gibi gibi ölçüler

overfit ve underfit
erken durdurma : earlystopping

AND GATE
INPUT 1     INPUT 2  OUTPUT
0           0        0
0           1        0
1           0        0
1           1        1
HER İKİ DÜĞME AÇIKSA LAMBA YANAR

OR GATE
INPUT 1     INPUT 2  OUTPUT
0           0        0
0           1        1
1           0        1
1           1        1

1 DÜĞME BİLE AÇIKSA IŞIK YANIYOR

NOT (değil) GATE
INPUT 1     OUTPUT
0           1
1           0

XOR (özel veya) GATE
INPUT 1     INPUT 2  OUTPUT
0           0        0
0           1        1
1           0        1
1           1        0

(0,0) ve (1,1) = 0 LAMBA SÖNÜK
(0,1) VE (1,0) = 1 LAMBA YANIK


AND = İKİSİDE 1
OR  = EN AZ BİRİ 1
NOT = 0 İSE 1
XOR = YALNIZCA 1'İ 1



öğrenci var :
x1 : 3 saat çalışıyor
x2 : 8 saat uyuyor
 geçti(1) kaldı(0)

w1 : 0.6
w2 : 0.2


z = (x1 x w1) + (x2 x w2) + b

step =>  z > 0 -> 1, z<= 0 -> 0
sigmoide => 0-1 arasında yumuşak bir değere dönüştürür.
reLU => z<0 -> 0, 0 negatif yok

bias -> ağın karar eşiğini belirler.

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,0,0,1])

w = np.random.rand(2)
b = np.random.rand(1) #bias

lr = 0.1 #öğrenme oranı (learning rate) ne kadar hızlı öğrensin ?

def step(z):
    return 1 if z >= 0.5 else 0

#100 tekrar epoch
for epoch in range(100):
    total_error = 0
    for i in range(len(X)):
        x1,x2 = X[i]
        z = x1 * w[0] + x2 * w[1] + b
        y_pred = step(z)

        error = y[i] - y_pred

        w[0] = w[0] + lr * error * x1
        w[1] = w[1] + lr * error * x2
        b = b + lr * error

        total_error += abs(error)

    if total_error == 0:
        break #hepsini doğru öğrenmiştir durrrr.
print("eğitim bitti....")
print("weight",w)
print("bias",b)

for i in range(len(X)):
    x1, x2 = X[i]
    z = x1 * w[0] + x2 * w[1] + b
    y_pred = step(z)
    print(f"girdi : {X[i]}, tahmin: {y_pred} , gerçek {y[i]}")



tek nöron (tek çizgi)
1 1
0 0

iki nöron (iki çizgi)
1|0
0|1


XOR = (x1 OR x2) AND NOT(x1 AND x2)
ikisi aynıysa 0, farklıysa 1
"""
import keras.src.ops
import numpy as np

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])


y = np.array([[0],[1],[1],[0]])

np.random.seed(42)

W1 = np.random.randn(2,2) # 2 giriş -> 2 hidden layer
b1 = np.zeros((1,2))
W2 = np.random.randn(2,1) # 2 gizli -> 2 çıkış
b2 = np.zeros((1,1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1-x)

lr = 0.5

for epoch in range(10000):
    z1 = np.dot(X, W1) + b1 #hidden layer girişi
    a1 = sigmoid(z1) #hidden layer çıkışı
    z2 = np.dot(a1, W2) + b2 #çıkış katman girişi
    a2 = sigmoid(z2) #ağın tahmini

    #hata (loss)
    error = y - a2

    #backprob.
    d_a2 = error * sigmoid_deriv(a2) # çıkış katmanı gradyanı
    d_a1 = d_a2.dot(W2.T) * sigmoid_deriv(a1) #gizli katman gradyanı

    #ağırlıkları güncelleme
    W2 += a1.T.dot(d_a2) * lr
    b2 += np.sum(d_a2,axis=0,keepdims=True) * lr
    W1 += X.T.dot(d_a1) * lr
    b1 += np.sum(d_a1,axis=0,keepdims=True) * lr

    #1000 adımda bir hatayı yazdıralım
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"epoch {epoch} : hata : {loss:.4f}")

print("tahminler")
print(a2.round(3))


(0,0) 0 0.017
(0,1) 1 0.49

model = keras.Sequential([
    layers.Dense(2,activation="sigmoid", input_shape(2,)),
    layers.Dense(1, activation="sigmoid")
])
layers.Dense(2,activation="tanh")

ö1 = 0.2 önem
ç2 = 0.5 önem
ö3 = 0.8 önem

model = keras.Sequential([
    layers.Dense(2,activation="tanh", input_shape(2,)),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam", #optimizer burda belirtilir
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#optimizer = en az hataya ulaşmak amacıyla optimize etmek.

her train adımında optimizer :
modelin hata oranına (loss) bakar
gradyan dediğimiz "hata eğimi" hesaplar.
ağırlıkları o yöne doğru küçük küçük günceller.


loss : amacım sayıyı sıfıra yaklaştırmak. hata miktarını ölçüyor.
loss mat. mantığı : (gerçek - tahmin)'2

tahmin -> loss hesapla -> optimizer düzelt -> yeni tahmin -> loss azalır.


model.fit(X,y,epochs=2000,batch_size=4)
model.evaluate(X,y)
model.predict(X)







