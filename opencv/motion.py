import cv2
import os
from datetime import datetime

cap = cv2.VideoCapture(0)
ret,frame1 = cap.read()
ret,frame2 = cap.read() #ikinci kare
save_dir = "kayitlar"
os.makedirs(save_dir,exist_ok=True)

while True:
    ret, frame1 = cap.read()
    ret2, frame2 = cap.read()  # ikinci kare

    if not ret or not ret2:
        print("kamera okunmadı..")
        break
    #iki kare arasında ki mutlak fark ?
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    #thresold : belirli bir farktan büyük yerleri beyaz yap.
    _, thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh,None,iterations=3)

    contours, _ = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hareket_var = False

    for c in contours:
        area = cv2.contourArea(c)
        if area < 20000:
            continue
        hareket_var = True
        x,y,w,h = cv2.boundingRect(c)

        pad = 20
        x1 = max(x - pad,0)
        y1 = max(y - pad, 0)
        x2 = min(x+w+pad, frame1.shape[1])
        y2 = min(y+h+pad, frame1.shape[0])

        crop = frame1[y1:y2, x1:x2]
        now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_dir, f"motion_{now}.jpg")
        cv2.imwrite(filename, crop)
        print("kesilen hareket kaydedildi...", filename)

        cv2.rectangle(frame1,(x,y), (x+w, y+h), (0,0,255),2)
        cv2.putText(frame1, "HAREKET!", (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    """
    if hareket_var:
        now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_dir,f"motion_{now}.jpg")
        cv2.imwrite(filename,frame1)
        print("hareket kaydedildi...",filename)
    """

    cv2.imshow("hareket algılama",frame1)
    cv2.imshow("maske",dilated)

    frame1 = frame2
    ret,frame2 = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()