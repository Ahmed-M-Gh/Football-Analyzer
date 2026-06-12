from ultralytics import YOLO

model = YOLO('model/yolo11n.pt') 

# exporting model to be more efficient for deployment
model.export(format='onnx', imgsz=640)