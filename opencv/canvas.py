import cv2
import numpy as np

cap = cv2.VideoCapture(0)

canvas = None

prev_point = None

lower_blue = np.array([100,150,50])
upper_blue = np.array([140,255,255])

while True:
    ret,frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,lower_blue,upper_blue)

    mask = cv2.GaussianBlur(mask,(5,5),0)
    mask = cv2.erode(mask,None,iterations=1)
    mask = cv2.dilate(mask,None,iterations=2)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_point = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 500:
            x,y,w,h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2
            current_point = (cx,cy)

            cv2.circle(frame,current_point,7,(0,0,255),-1)

        if current_point is not None:
            if prev_point is not None:
                cv2.line(canvas,prev_point,current_point,(0,255,0),5)
            prev_point = current_point
        else:
            prev_point = None

        combined = cv2.addWeighted(frame,0.7,canvas,0.8,0)

        cv2.imshow("maske",mask)
        cv2.imshow("tahta",combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()















