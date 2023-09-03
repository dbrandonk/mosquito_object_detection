"""Mosquito yolo classification training."""

from pathlib import Path
from ultralytics import YOLO

# This seems jank. Haven't tried hard enough to find a better way.
PACKAGE_PATH = Path(__file__).resolve().parent.parent.parent


def main():
    """main"""
    # load a pretrained model
    model = YOLO(PACKAGE_PATH / Path('weights/yolov8m-cls.pt'))

    # train
    model.train(data=(PACKAGE_PATH / Path('data/mosquito_alert_2023/classification')), epochs=300, imgsz=640)


if __name__ == "__main__":
    main()
