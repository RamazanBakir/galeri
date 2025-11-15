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
"""





