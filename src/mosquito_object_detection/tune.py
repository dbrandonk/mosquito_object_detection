"""Mosquito object detection yolo tuning."""

from pathlib import Path
from ultralytics import YOLO

# This seems jank. Haven't tried hard enough to find a better way.
PACKAGE_PATH = Path(__file__).resolve().parent.parent.parent

# load a pretrained model
model = YOLO(PACKAGE_PATH / Path('weights/yolov8n.pt'))

# tune
model.tune(data=(PACKAGE_PATH / Path('config/mosquito_alert.yaml')), grace_period=15, max_samples=20, epochs=100)
