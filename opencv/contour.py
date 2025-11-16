import cv2
"""
konturu bul 
konturu daha az noktaya indiriyoruz (approxPolyDP)
kaç köşe var ? şekli belirlemek için
3 köşe var -> üçgen 
4 köşe var -> dörtgen
5 köşe -> beşgen 
çok fazla köşe varsa -> daireye yakın (oval)

img = cv2.imread("karpuz2.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

counturs,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,counturs,-1,(0,255,0),2)

cv2.imshow("kontr",img)
cv2.imshow("thresh",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

approx = cv2.approxPolyDP(counturs,epsilon,True)
"""

img = cv2.imread("karpuz.jpeg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

_,tresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

contours,_ = cv2.findContours(tresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue #çok küçük olan şeyleri göz ardı et

    peri = cv2.arcLength(cnt,True)

    approx = cv2.approxPolyDP(cnt,0.02*peri,True)

    corners = len(approx)

    x,y,w,h = cv2.boundingRect(approx)


    if corners == 3 :
        shape = "ucgen"
    elif corners == 4:
        aspect = w / float(h)
        shape = "kare" if 0.95 < aspect < 1.05 else "dikdortgen"
    elif corners == 5:
        shape = "besgen"
    else:
        shape = "daire"

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img,shape,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
cv2.imshow("sekil tespiti", img)
cv2.imshow("thres",tresh)
cv2.waitKey(0)
cv2.destroyAllWindows()












