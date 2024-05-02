import ultralytics
import os
ultralytics.checks()

current_path = os.getcwd()
print(current_path)

model = ultralytics.YOLO('yolov8m.pt')
yaml_path = 'Create_model\Sulivan_train\data\data.yaml'

model.train(data=yaml_path, epochs=300, batch=40, imgsz=500)