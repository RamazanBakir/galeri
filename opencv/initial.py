import cv2
import numpy as np
"""
cv2.imread("resim.jpg") #resim okur
cv2.imshow(...) #resmi gösterir
siyah beyaz, kenarları bulmak için, yüz tanıma için,

img = cv2.imread("karpuz.jpeg")

if img is None:
    print("resim bulunamadı...")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("orjinal resim boyutu, (yükseklik,genişlik...)",img.shape)
    print("gri resim boyutu, (yükseklik,genişlik...)",gray.shape)
    cv2.imshow("resim karpuz",img)
    cv2.imshow("resim karpuz gri",gray)

    cv2.waitKey(0) #herhangi bir tuşa basılana kadar bekle.
    #waitkey örn. 1000 = 1 saniye bekler sonra kapatır
    #açık pencereli kapat
    cv2.destroyAllWindows()

bgr (blue, green,red)
ortalama blur (average blur) - en sade yöntem
gaussaian blur - en kaliteli ve en çok kullanılan
median blur - gürültü temizlemede kral 

img = cv2.imread("karpuz.jpeg")

blur = cv2.blur(img,(3,3))
blur1 = cv2.blur(img,(5,5))
blur2 = cv2.blur(img,(15,15))
blur3 = cv2.blur(img,(30,30))
gauss = cv2.GaussianBlur(img,(5,5),0)
gauss2 = cv2.GaussianBlur(img,(5,5),5)
median = cv2.medianBlur(img,5) #window size tek sayı olmak zorunda (3,5,7,9...)


cv2.imshow("orjinal",img)
cv2.imshow("blur",blur)
cv2.imshow("gaus",gauss)
cv2.imshow("median",median)

cv2.imshow("gauss2",gauss2)
cv2.imshow("blur1",blur1)
cv2.imshow("blur2",blur2)
cv2.imshow("blur3",blur3)


cv2.waitKey(0)
cv2.destroyAllWindows()

kenar = renk veya parlaklık değişiminin olduğu yer.



img = cv2.imread("karpuz.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(5,5),0)
edges = cv2.Canny(gray,100,200)
edges2 = cv2.Canny(blur,100,200)

cv2.imshow("resim karpuz", img)
cv2.imshow("kenarlar",edges)
cv2.imshow("kenarlar blur",edges2)

cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread("karpuz2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gri yaptık
_, thresh = cv2.threshold(gray, 127,255,cv2.THRESH_BINARY) #threshold
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #kontur bul
cv2.drawContours(img, contours, -1, (0,255,0),2)

cv2.imshow("kontur",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

area = cv2.contourArea(contours) #alanını bul
big = max(contours,key=cv2.contourArea()) #en büyük nesneyi bul...
M = cv2.moments(contours)
cx = int(M["m10"]/M["m00"])
cy = int(M["m01"]/M["m00"])

cv2.circle(img,(cx,cy),5,(0,0,255),-1)

img = cv2.imread("karpuz2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gri yaptık
_, thresh = cv2.threshold(gray, 127,255,cv2.THRESH_BINARY) #threshold
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #kontur bul
cnt = contours[0]
peri = cv2.arcLength(cnt,True)
epsilon = 0.02 * peri

approx = cv2.approxPolyDP(cnt,epsilon,True)


print(len(contours))
big_cnt = max(contours, key = cv2.contourArea)
print("en büyük kontur alanı",cv2.contourArea(big_cnt))

peri = cv2.arcLength(big_cnt,True) #çevre
for f in [0.001, 0.005, 0.01, 0.02]:
    epsilon = f * peri
    approx = cv2.approxPolyDP(big_cnt,epsilon,True)
    print(f,"köşe sayısı", len(approx))
cv2.drawContours(img,[approx],-1,(0,255,0),2)
cv2.imshow("karpuz kontorrr",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread("karpuz2.jpg")
if img is None:
    print("resim yok")
else:
    print("orj. boyut",img.shape) #yükseklik,genişlik,kanal

    h,w = img.shape[:2]

    new_w = w // 2
    new_h = h // 2
    resized = cv2.resize(img,(new_w,new_h))
    print("yeni boyut",resized.shape)

    cv2.imshow("orj.",img)
    cv2.imshow("resized",resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
parca = img[y1:y2, x1:x2]
sıra [Y,X] -> [satır,sütun] -> [yükseklik, genişlik]


img = cv2.imread("karpuz2.jpg")
if img is None:
    print("resim yok")
else:
    h,w = img.shape[:2]
    print("boyut", h, w)

    #merkezin etrafında 200x200lük kare alalım
    box_size = 200
    center_y = h //2
    center_x = h // 2

    y1= center_y - box_size // 2
    y2 = center_y + box_size // 2
    x1 = center_x - box_size // 2
    x2 = center_x + box_size // 2

    crop = img[y1:y2, x1:x2]
    cv2.imshow("orj",img)
    cv2.imshow("crop",crop)
    #yukarıdan aşağıya soldan sağa gibi düşünelim
    cv2.waitKey(0)
    cv2.destroyAllWindows()

3 temel çizim fonks: 
cv2.line -> çizgi
cv2.rectangle -> dikdörtgen
cv2.circle -> daire
renkler : bgr
örn: 
BGR(BLUE,GREEN,RED)
(255,0,0)
(0,255,0)
(0,0,255)

cv2.line(img,(x1,y1), (x2,y2), (B,G,R), kalinlik)
cv2.rectange(img,(x1,y1), (x2,y2), (B,G,R), kalinlik)
cv2.circle(img,(merkez_X,merkez_Y),yaricap, (B,G,R), kalinlik)

cv2.putText(
    draw_img,
    "yazi yaz",
    (x,y),
    font,
    font_scale,
    (B,G,R),
    kalinlik,
    lineType
)

img = cv2.imread("karpuz2.jpg")
draw_img = img.copy()
cv2.line(draw_img,(50,50),(300,50),(0,0,255),3)
cv2.rectangle(draw_img,(100,100),(300,300),(0,255,0),2)
cv2.circle(draw_img, (200,200),50,(255,0,0),3)

text_img = img.copy()
cv2.putText(
    text_img,
    "hello class",
    (50,50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0,255,255),
    2,
    cv2.LINE_AA
)
cv2.imshow("çizgi çizilen",draw_img)
cv2.imshow("text yazılan",text_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#webcam ile gerçek zamanlı görüntü işleme...


face = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

img = cv2.imread("pep.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)

cv2.imshow("sonuc",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


print(cv2.data.haarcascades)

cap = cv2.VideoCapture(0) #0 : varsayılan kamera

if not cap.isOpened():
    print("kamera açılmadı")
else:
    print("kamera açıldı...")
    while True:
        ret, frame = cap.read()

        if not ret:
            print("kare okunamadı, çıkılıyor...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray,(5,5),0)

        edges = cv2.Canny(blurred,50,150)

        cv2.imshow("kamera",frame)
        cv2.imshow("gri",gray)
        cv2.imshow("edges",edges)



        #q tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
(x,y,w,h) -> şu bölgede yüz var...
face recognition -> yüz kime ait ?
çıktı : etiket var. 

faces = face.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30)
)


face = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_eye.xml")
smile = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_smile.xml")

#img = cv2.imread("pep.png")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("kamera açılmadı")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("kareyi bulamadım")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, 1.1, 5,minSize=(100,100))
        for (x, y, w, h) in faces:
            #yüz dikdörtgeni
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w] #gri yüz bölgesi
            roi_color = frame[y:y+h,x:x+w] #renkli yüz bölgesi

            eyes = eye.detectMultiScale(roi_gray,1.1,5,minSize=(20,20))
            for  (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey), (ex + ew, ey + eh), (255,0,0),2)

            smiles = smile.detectMultiScale(roi_gray, scaleFactor=1.7,minNeighbors=22,minSize=(25,25))
            for  (sx,sy,sw,sh) in smiles:
                cv2.rectangle(roi_color,(sx,sy), (sx + sw, sy + sh), (0,255,255),2)

            if len(smiles) > 0:
                cv2.putText(
                    frame,
                    "Gulumseme oldu :)",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,255),
                    2
                )

        cv2.putText(
            frame,
            f"yüz sayısı: {len((faces))}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0,255,0),
            2
        )
        cv2.imshow("sonuc", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
RGB : RED, GREEN, BLUE
opencv de şöyle:  BGR -> BLUE,GREEN,RED -> (10,200,50)

HSV (Hue, Saturation, Value)
Hue : ana renk : 0-180 arasında (opencv) 0:kırmıza, 60:sarı, 120:yeşil, 180: mavi tonlarına gider
Saturation: rengin doygunluğu (soluk/canlı)
Value: parlaklık (karanlık mı / aydınlık mı)

resmi frame vs. oku (bgr formatında gelir)
cvtColor ile : BGR -> HSV çevir
lower ve upper değeri belirle
inRange le maske oluştur
bu maske ile sadece o rengi göster (bitwise_and)

img = cv2.imread("karpuz.jpeg")

if img is None:
    print("resim yok")
else:
    #BGR->HSV'ye çevir.
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,120,70]) #[H,S,V]
    upper_red = np.array([10,255,255])

    lower_red1 = np.array([0,120,70]) #[H,S,V]
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,120,70]) #[H,S,V]
    upper_red2 = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red1,upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2,upper_red2)
    mask = mask1 | mask2

    #bu aralıktaki pikselleri 1, diğerlerini 0 yap
    #mask = cv2.inRange(hsv, lower_red,upper_red)
    #mask ile sadece kırmızı olanları göster
    result = cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow("orjin",img)
    cv2.imshow("maske(siyah-beyaz)",mask)
    cv2.imshow("sadece kırmızı", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("kamera açılmadı")
else:
    while True:
        ret,frame = cap.read()
        if not ret:
            print("kareyi okuyamadım")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0,120,70]) #[H,S,V]
        upper_red1 = np.array([10,255,255])

        lower_red2 = np.array([170,120,70]) #[H,S,V]
        upper_red2 = np.array([180,255,255])

        mask1 = cv2.inRange(hsv, lower_red1,upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2,upper_red2)
        mask = mask1 | mask2

        result = cv2.bitwise_and(frame,frame,mask=mask)

        cv2.imshow("orjin",frame)
        cv2.imshow("maske(siyah-beyaz)",mask)
        cv2.imshow("sadece kırmızı", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()











