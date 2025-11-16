import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    h,w, _ = frame.shape
    roi_size = 300
    cx, cy = w // 2, h //2
    roi_x1 = cx - roi_size // 2
    roi_y1 = cy - roi_size // 2
    roi_x2 = cx + roi_size // 2
    roi_y2 = cy + roi_size // 2

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    lower_skin = np.array([0,135,85], dtype=np.uint8)
    upper_skin = np.array([255,180,135], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower_skin,upper_skin)

    kernel = np.ones((3,3),np.uint8)
    mask = cv2.GaussianBlur(mask,(5,5),0)
    mask = cv2.erode(mask,kernel,iterations=1)
    mask = cv2.dilate(mask,kernel,iterations=2)

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finger_count = 0

    if len(contours) > 0:
        c = max(contours,key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 1000:
            cv2.drawContours(roi,[c],-1,(0,255,0),2)

            hull = cv2.convexHull(c,returnPoints=False)

            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(c,hull)

                valid_defects = 0

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(c[s][0])
                        end = tuple(c[e][0])
                        far = tuple(c[f][0])

                        a = math.dist(start,end)
                        b = math.dist(start,far)
                        c_len = math.dist(end,far)

                        if b != 0 and c_len != 0 :
                            cos_angle = (b**2 + c_len**2 - a**2) / (2 * b * c_len)
                            cos_angle = max(-1.0, min(1.0,cos_angle))
                            angle = math.degrees(math.acos(cos_angle))

                            depth = d / 256.0

                            if angle < 90 and depth > 20:
                                valid_defects += 1
                                cv2.circle(roi,far, 5,(0,0,255),-1)
                    finger_count = valid_defects + 1

                    if finger_count > 5:
                        finger_count = 5

                    if area < 8000 and finger_count > 1 :
                        finger_count = max(0,finger_count -1)

    frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2,roi_y2), (255,0,0), 2)
    cv2.putText(frame,"elini buraya getir",(roi_x1 - 90, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),2)

    cv2.putText(frame,f"parmak: {finger_count}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)

    cv2.imshow("kamera",frame)
    cv2.imshow("mask",mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




















