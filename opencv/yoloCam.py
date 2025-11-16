from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")
names = model.names
cap = cv2.VideoCapture(0)

person_id = None
for cls_id, cls_name in names.items():
    if cls_name == "person":
        person_id = cls_id
        break
print("person class id",person_id)


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

    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls_id != person_id:
                continue
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2 ,y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

            label = f"{names[cls_id]} {conf:.2f}"

            cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()