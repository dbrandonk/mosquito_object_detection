from pathlib import Path
from ultralytics import YOLO

dir_path = Path(__file__).resolve().parent
# dir_path = os.path.dirname(os.path.abspath(__file__))

# load a pretrained model
model = YOLO(dir_path / Path('../../checkpoints/yolov8n.pt'))

model.train(data=(dir_path / Path('../../config/mosquito_alert.yaml')), epochs=100, imgsz=640)
