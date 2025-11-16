from ultralytics import YOLO

#hazır model yükledik
model = YOLO("yolov8n.pt")

#tahmin yaptıralım
result = model("karpuz.jpeg")

result = result[0]

classes = result.boxes.cls

scores = result.boxes.conf

names = model.names

for cls, conf in zip(classes,scores):
    cls = int(cls.item())
    conf = float(conf.item())
    label = names[cls]
    print(f"nesne : {label}, güven : {conf:.2f}")

result[0].save(filename="sonuc2.jpg")