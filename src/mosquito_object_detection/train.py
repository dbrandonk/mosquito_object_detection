"""Mosquito object detection yolo training."""

from pathlib import Path
from ultralytics import YOLO

# This seems jank. Haven't tried hard enough to find a better way.
PACKAGE_PATH = Path(__file__).resolve().parent.parent.parent


def main():
    """main"""
    # load a pretrained model
    model = YOLO(PACKAGE_PATH / Path('weights/yolov8m.pt'))

    data_aug_args = {
        'copy_paste': 0.4265190235061216,
        'degrees': 33.027968973162224,
        'fliplr': 0.14544711724779424,
        'flipud': 0.6130873162856098,
        'hsv_h': 0.044431462154359204,
        'hsv_s': 0.37653030080836375,
        'hsv_v': 0.6556806225166533,
        'mixup': 0.4001527404044749,
        'mosaic': 0.5929641148016892,
        'perspective': 2.288614373103448e-05,
        'scale': 0.48552095776791077,
        'shear': 4.727303905928761,
        'translate': 0.23874463557515493}

    # train
    model.train(data=(PACKAGE_PATH / Path('config/mosquito_alert.yaml')),
                batch=16,
                close_mosaic=25,
                epochs=300,
                imgsz=640,
                lr0=0.001,
                optimizer='Adam',
                momentum=0.937,
                patience=50,
                pretrained=True,
                **data_aug_args)


if __name__ == "__main__":
    main()
