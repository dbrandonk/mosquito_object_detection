"""Mosquito object detection yolo training."""

from pathlib import Path
from ultralytics import YOLO

# This seems jank. Haven't tried hard enough to find a better way.
PACKAGE_PATH = Path(__file__).resolve().parent.parent.parent

# load a pretrained model
model = YOLO(PACKAGE_PATH / Path('weights/yolov8n.pt'))

# train
model.train(data=(PACKAGE_PATH / Path('config/mosquito_alert.yaml')),
            batch=16,
            close_mosaic=10,
            epochs=300,
            imgsz=640,
            lr0=0.001,
            optimizer='Adam',
            momentum=0.937,
            patience=50,
            pretrained=True)
