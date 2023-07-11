"""Mosquito object detection yolo training."""

from pathlib import Path
from ultralytics import YOLO

# This seems jank. Haven't tried hard enough to find a better way.
package_path = Path(__file__).resolve().parent.parent.parent

# load a pretrained model
model = YOLO(package_path / Path('checkpoints/yolov8n.pt'))

model.train(data=(package_path / Path('config/mosquito_alert.yaml')), epochs=100, imgsz=640)
