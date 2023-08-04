"""Mosquito object detection yolo tuning."""

from pathlib import Path
from ray import tune
from ultralytics import YOLO

# This seems jank. Haven't tried hard enough to find a better way.
PACKAGE_PATH = Path(__file__).resolve().parent.parent.parent


def main():
    """main"""
    # load a pretrained model
    model = YOLO(PACKAGE_PATH / Path('weights/yolov8n.pt'))

    search_space = {
        'hsv_h': tune.uniform(0.0, 0.1),
        'hsv_s': tune.uniform(0.0, 0.9),
        'hsv_v': tune.uniform(0.0, 0.9),
        'degrees': tune.uniform(0.0, 45.0),
        'translate': tune.uniform(0.0, 0.9),
        'scale': tune.uniform(0.0, 0.9),
        'shear': tune.uniform(0.0, 10.0),
        'perspective': tune.uniform(0.0, 0.001),
        'flipud': tune.uniform(0.0, 1.0),
        'fliplr': tune.uniform(0.0, 1.0),
        'mosaic': tune.uniform(0.0, 1.0),
        'mixup': tune.uniform(0.0, 1.0),
        'copy_paste': tune.uniform(0.0, 1.0)}

    train_args = {
        'batch': 16,
        'close_mosaic': 10,
        'epochs': 300,
        'imgsz': 640,
        'lr0': 0.001,
        'optimizer': 'Adam',
        'momentum': 0.937,
        'patience': 50,
        'pretrained': True}

    # tune
    model.tune(data=(PACKAGE_PATH / Path('config/mosquito_alert.yaml')),
               gpu_per_trial=1,
               grace_period=10,
               max_samples=5,
               space=search_space,
               **train_args
               )


if __name__ == "__main__":
    main()
