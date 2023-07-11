"""Mosquito object detection yolo training."""

from pathlib import Path
from ultralytics import YOLO

# This seems jank. Haven't tried hard enough to find a better way.
PACKAGE_PATH = Path(__file__).resolve().parent.parent.parent

# load a pretrained model
model = YOLO(PACKAGE_PATH / Path('checkpoints/yolov8n.pt'))

# train
model.train(data=(PACKAGE_PATH / Path('config/mosquito_alert.yaml')), epochs=100, imgsz=640)
