import cv2
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
"""
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





















