from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("kamera açılmadı")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        print("kare okuyamadım")
        break

    result = model(frame)
    result = result[0]

    annoted = result.plot()

    cv2.imshow("kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()