import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
[1,2,3,4,5,6]

yeni = []
for x in liste:
    yeni.append()

liste = [1,2,3,4,5,6]

a = np.array([1,2,3,4,5,6])
print(a)
print(a*2)
print(a + 10)
print(a / 2)


b = np.array([
    [1,2,3],
    [4,5,6]
])
print(b)
print(b.shape) #boyutunu gösterir (2,3) 2 satır, 3 sütun
#slicing
print(b[0]) # ilk satır
print(b[1,2]) #2.satır, 3.sütun
print(b[:,1]) #tüm satırların 2.sütunu

a = np.array([5,10,15,20,25])
print(a > 12)
print(a[a>12])

1-)
a = [3,6,9,12,15]
numpy array -> tüm elemanlarını 2 katına çıkar
2-) numpy dizisi oluşturun
1 2 3
4 5 6
7 8 9
orta elemanı seç
sadece son sütunu seçin
10'dan büyük olan var mı kontrol et.

a = np.array([1,2,3,4,5,6])
print(a)
print(type(a))

b = np.array([[1,2], [3,4], [5,6]])
print(b.shape)

x = np.arange(0,10)
y = np.arange(0,10,2) #0dan 10a kadar 2şer
print(y)

print(np.zeros((3,3)))

print(np.ones((2,5)))

print(np.random.rand(3,3)) #0-1 arası
print(np.random.randint(0,10,(3,3))) #0 - 10 arası tam sayılar

a = np.array([10,20,30,40])
print(a[0])
print(a[-1])

b = np.array([[10,20,30],[40,50,60]])
print(b)
print(b[0,1])
print(b[1,2])

a = np.array([10,20,30,40,50])
print(a[1:4])
print(a[:3])
print(a[::2]) # [10 30 50]
#a[start : stop : step]

a = [1,2,3,4]
print(type(a))
print(a * 2)


b = np.array([1,2,3,4,5,6])
print(type(b))
print(b * 2 )


a = np.array([1,2,3])
print(a +10)
print(a *2)
print(a **2)

b = np.array([10,20,30])
print(a+b)

#MATRİS ÇARPIMI = A'NIN SATIRI * B'NİN SÜTUNU
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
print(A)
print(B)

print(A @ B)

data = np.array([0,10,20,30,40])
print(data.mean()) #ortalama
print(data.sum()) #toplam
print(data.max())
print(data.min())
print(data.std()) #standart sapma


x = np.arange(1,12)
print(x)
#print(x.reshape(3,4))
print(x.reshape(-1,1))

#transpoze 
satırlar -> sütun
sütun -> satır


A = [1 2 3
     4 5 6 ]

At = [1 4
     2 5
     3 6]
A = np.array([[1,2,3],[4,5,6]])
print(A)
A = A.T
print(A)
#broadcasting

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([10,20,30])
print( A + b)
print(A + 3)

a = np.array([1,2,3,4])
b = np.array([4,5,6])
c = np.concatenate([a,b])
print(c)

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
print(x)
print(y)
#satır bazında (altına ekle)
z1 = np.concatenate([x,y],axis=0)
print(z1)
#sütun bazında (yanına ekle)
z2 = np.concatenate([x,y],axis=1) 
print(z2)

#vertical -> dikey, horizontal -> yatay
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.vstack([a,b]))
print(np.hstack([a,b]))

x = np.array([10,20,30,40,50])
print(np.where(x > 20))

y = np.where(x>25,x,0)
print(y)
print(x[x>25])


nums = np.array([1,1,2,2,2,2,3,4,5,5,5,6,6,6,6,6,6,6,4,4,4,1,2,8,8])
vals,counts = np.unique(nums,return_counts=True)
print(vals) #tekrarsız değerler
print(counts) #her birinin adedi

#np.linalg -> matris matematiği

A =np.array([[1,2],[3,4]])
det = np.linalg.det(A)
print(A)
print(det)
#invers (matrisin tersi) -> A* A(-1) = I 
inv = np.linalg.inv(A)
print(inv)


[a b 
 c d]  1/(ad- bc) * [d -b
                     -c a]

[2 3
 1 4]
(2*4 - 3*1) = 5 #determinant

inv = 1/5 * [4 -3
             -1 2]
[4/5 -3/5
 -1/5 2/5]

A =np.array([[2,3],[1,4]])
print(A)
print(np.linalg.det(A))
print(np.linalg.inv(A))

A =np.array([[1,2],[3,4]])
B =np.array([[5,6],[7,8]])
print(A @ B ) #önerilendir
print(np.matmul(A,B)) #alternatif
print(np.linspace(0,1,5))
np.random.seed(42)
print(np.random.randint(0,10,5))


notlar = np.array([55,70,85,90,40,60,-1])

print(notlar.mean())
print(notlar[notlar >=60])
print(notlar[notlar < 60])
print((notlar < 0).any())

r = [3,6,9,12,15]
print(r)
print(type(r))

x = np.array(r)
print(x)
print(type(x))
print((x+5)/3)

1 2 3
4 5 6
7 8 9


A = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print(A[1,1])
print(A[:,2])
print(A[A > 4])


# ilk veri yapısı : DataFrame = Satır + sütun görüntü (excel)

data = {
    "ad": ["hatice","ahsen","nisa","ramazan","nuray","ahmet"],
    "yas": [20,22,24,30,21,33],
    "sehir":["ankara","istanbul","izmir","mugla","antalya","antep"],
    "boy": [220, 222, 224, 230, 221, 233],
}
print(data)
df= pd.DataFrame(data)#dataframe oluşturma
print(df)
print(df.head()) #head(10) diye sayı verebilirsin.
print(df.info())
print(df.describe())
print("-"*100)
#sütun seç
print(df["ad"])
print(df["yas"])
print(df[["ad","sehir"]])

data = {
    "ad": ["hatice","ahsen","nisa","ramazan","nuray","ahmet"],
    "yas": [20,22,24,17,21,33],
    "sehir":["ankara","istanbul","izmir","mugla","antalya","antep"],
    "boy": [220, 222, 224, 230, 221, 233],
}
print(data)
df= pd.DataFrame(data)#dataframe oluşturma
print(df)

#satır seç
#loc (etiketle) ve iloc (index numarasına göre)
#print(df.iloc[0]) #0.satır
#print(df.iloc[2]) #2.satır
print(df.iloc[0:3])
print(df.iloc[:,0])
print(df.iloc[1,2])

print(df.loc[0])
print(df.loc[0:3]) #0,1,2,3 dahil (loc -> bitişi dahil)
print("-"*100)
print(df.iloc[0:3])
print(df.loc[0:2,["ad","sehir"]])
print(df.loc[3,"sehir"])
print(df[(df["yas"] >= 20) & (df["sehir"] == "izmir")])
df["yetişkin mi?"] = df["yas"] >=18
print(df)
#dataframe'de silme işlemi (satır ya da sütun)
df = df.drop(columns=["boy"])
print(df)
df = df.drop(index=[5])
print(df)

df = pd.read_csv("veriler.csv")
print(df.head())
df["yetişkin mi?"] = df["yas"] >=18
print(df)
df.to_csv("cikti.csv",index=False)

data = {
    "ürün":["su","ekmek","domates","karpuz","kavun","limon"],
    "fiyat": [5,6,7,1,3,10],
    "adet": [3,1,2,6,2,1]
}

df = pd.DataFrame(data)
print(df)

df["Toplam"] = df["fiyat"] * df["adet"]
print(df)
print("genel toplam:",df["Toplam"].sum())
df.to_csv("cikti.csv",index=False)

df = pd.read_csv("veriler.csv")
print(df.head())
print(df.isna().sum())

df2 = df.copy()
df2["yas"] = df2["yas"].fillna(df2["yas"].median()) #medyan
df2["boy"] = df2["boy"].fillna(df2["boy"].mean()) #ortalama
print(df2)
#ffil ve a
#df2["tarih"] = df2["tarih"].ffill() #tarihi bir önceki satırla doldurur
df2["tarih"] = df2["tarih"].bfill() #tarihi bir sonraki satırla doldurur

print(df2)
#print(df2["sehir"].str.title())
#print(df2["sehir"].str.upper())
#print(df2[df2["sehir"].str.startswith("iğ")]) #sadece başında arama yapar.
#print(df2[df2["sehir"].str.contains("ta",case=False,regex=False)]) #içinde geçen

df2["tarih"] =pd.to_datetime(df2["tarih"])
df2["yil"] = df2["tarih"].dt.day
print(df2)

x = [1,2,3,4] #x ekseni
y = [10,20,30,40]

#plt.plot(x,y) #çizgi çiz
#plt.bar(x,y) #çubuk grafik
plt.scatter(x,y) #nokta grafiği
plt.show()
sehirler = ["muğla","istanbul","izmir"]
nufus = [100,300,200]

plt.bar(sehirler,nufus)
plt.show()

degerler = [40,30,20,10]
etiketler = ["A","B","C","D"]

plt.pie(degerler,labels=etiketler)
plt.show()



ogrenciler = ["Ali","Ahmet","Ayşe","Osman","Hasan"]
notlar = [70,85,90,25,69]

plt.bar(ogrenciler,notlar)
plt.title("matematik notları")
plt.xlabel("öğrenciler")
plt.ylabel("notlar")
plt.grid(True) #ızgara (grid) çizgisi ekler
plt.show()


fig, ax = plt.subplots(figsize=(6,4), dpi=120) #tek eksen
ax.plot([1,2,3,4],[10,20,25,40])
ax.set_title("Satış Trendi")
ax.set_xlabel("Gün")
ax.set_ylabel("Adet")
ax.grid(True,alpha=0.3)
fig.tight_layout()
fig.savefig("satis_trend.pdf")

"""



