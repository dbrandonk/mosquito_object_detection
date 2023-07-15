"""Mosquito object detection yolo prediction."""

from pathlib import Path
import csv
import shutil

from ultralytics import YOLO

# This seems jank. Haven't tried hard enough to find a better way.
PACKAGE_PATH = Path(__file__).resolve().parent.parent.parent

TEST_IMAGE_PATH = PACKAGE_PATH / Path('data/mosquito_alert_2023/images/test')
MODEL_PATH = PACKAGE_PATH / Path('weights/mosquito_od_yolov8n.pt')
SUBMISSION_FILE = PACKAGE_PATH / Path('mosquito_od_submission.cvs')


with open(SUBMISSION_FILE, 'w', encoding='utf-8') as csv_file:

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(['img_fName', 'img_w', 'img_h', 'bbx_xtl',
                        'bbx_ytl', 'bbx_xbr', 'bbx_ybr', 'class_label'])

    # load trained model
    model = YOLO(MODEL_PATH)

    image_files = shutil.os.listdir(TEST_IMAGE_PATH)

    for image_file in image_files:
        # predict
        result = model.predict(TEST_IMAGE_PATH / Path(image_file), max_det=1)

        # gather results

        xtl, ytl, xbr, ybr = result[0].boxes[0].xyxy[0].tolist()
        class_label = result[0].names[int(result[0].boxes[0].cls)]
        height, width = result[0].orig_shape
        csv_writer.writerow([image_file, width, height, xtl, ytl, xbr, ybr, class_label])
