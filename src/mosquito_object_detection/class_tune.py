"""Mosquito yolo classification tuning."""

from pathlib import Path
from ultralytics import YOLO

# This seems jank. Haven't tried hard enough to find a better way.
PACKAGE_PATH = Path(__file__).resolve().parent.parent.parent


def main():
    """main"""
    # load a pretrained model
    model = YOLO(PACKAGE_PATH / Path('weights/yolov8s-cls.pt'))

    # tune
    model.tune(data=(
        PACKAGE_PATH /
        Path('data/mosquito_alert_2023/classification')),
        gpu_per_trial=1,
        grace_period=10,
        iterations=8,
        use_ray=True,
        epochs=20
    )


if __name__ == "__main__":
    main()
