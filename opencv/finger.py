import cv2
import numpy as np
#roi bölgesi : region of interest
"""
            cv2.rectangle(frame,
                          (roi_x1 + x, roi_y1 + y),
                          (roi_x1 + x + w, roi_y1 + y + h),
                          (0,255,0),2)
"""
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #aynada kendimizi görür gibi olsun diye görüntüüy yatay çeviriyoruz
    frame = cv2.flip(frame,1)

    h,w,_ = frame.shape
    roi_size = 300
    roi_width,roi_height = roi_size,roi_size #kutu boyutu
    roi_x1 = w // 2 - roi_width // 2
    roi_y1 = h // 2 - roi_height // 2
    roi_x2 = roi_x1 + roi_width
    roi_y2 = roi_y1 + roi_height

    roi_x1 = max(0,roi_x1)
    roi_y1 = max(0,roi_y1)
    roi_x2 = min(w, roi_x2)
    roi_y2 = min(h, roi_y2)


    #roi bölgesi -sağ üst köşe
    #koordinat : y1:y2 x1:x2
    #roi_y1, roi_y2 = 50,300
    #roi_x1, roi_x2 = 350,600

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0,20,50])
    upper_skin = np.array([20,255,255])

    mask = cv2.inRange(hsv,lower_skin,upper_skin)

    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.erode(mask,None,iterations=1)
    mask = cv2.dilate(mask,None,iterations=2)

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours,key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 1000:
            cv2.drawContours(roi,[c],-1,(0,255,0),2)

            #convex hull hesaplama
            hull = cv2.convexHull(c)
            cv2.drawContours(roi,[hull],-1,(0,0,255),3)

            #x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame,(roi_x1,roi_y1),(roi_x2,roi_y2), (255,0,0),2)

            cv2.putText(frame,"EL", (roi_x1,roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0), 4)

    cv2.rectangle(frame,(roi_x1,roi_y1),(roi_x2,roi_y2),(255,0,0),2)
    cv2.putText(frame,"elini buraya getir",(roi_x1 - 50, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),4)

    cv2.imshow("kamera",frame)
    cv2.imshow("maske",mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








