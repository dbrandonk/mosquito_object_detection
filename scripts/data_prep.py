"""A very brittle script that setups up the data (https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023) nicely to use with YOLO."""  # pylint: disable=line-too-long

from pathlib import Path
import argparse
import os
import zipfile

import pandas as pd
import yaml

PACKAGE_PATH = Path(__file__).resolve().parent.parent
DATA_ROOT_PATH = PACKAGE_PATH / Path('data')
DATA_PATH = DATA_ROOT_PATH / Path('mosquito_alert_2023')
IMAGE_PATH = DATA_PATH / Path('images')
LABEL_PATH = DATA_PATH / Path('labels/train')
CLASS_MAPPING_PATH = PACKAGE_PATH / Path('config/mosquito_alert.yaml')
TRAINING_DATA_PATH = DATA_PATH / Path('train.csv')


def _convert_yolo_format(data_frame, class_mapping):

    for _, row in data_frame.iterrows():

        class_label = class_mapping[row['class_label']]
        x_center = float(row['bbx_xbr'] + row['bbx_xtl']) / (2 * row['img_w'])
        y_center = float(row['bbx_ybr'] + row['bbx_ytl']) / (2 * row['img_h'])
        width = float(row['bbx_xbr'] - row['bbx_xtl']) / row['img_w']
        height = float(row['bbx_ybr'] - row['bbx_ytl']) / row['img_h']

        with open(LABEL_PATH / Path(row['img_fName'].split('.')[0] + '.txt'),
                  'w', encoding='utf-8') as file:
            file.write(f'{class_label} {x_center} {y_center} {width} {height}\n')


def _get_class_mapping():

    try:
        with open(CLASS_MAPPING_PATH, 'rb') as file:
            yaml_data = yaml.safe_load(file)
        classes = {value: key for key, value in yaml_data['names'].items()}
    except FileNotFoundError:
        print("File not found.")

    return classes


def _read_training_data():
    try:
        data_frame = pd.read_csv(TRAINING_DATA_PATH)
    except FileNotFoundError:
        print("File not found.")

    return data_frame


def _get_data(api_key):
    login = f'aicrowd login --api-key {api_key}'
    os.system(login)

    ai_crowd_cmd = 'aicrowd dataset download --challenge mosquitoalert-challenge-2023'
    data_download = f'cd {str(DATA_PATH)}; {ai_crowd_cmd}'
    os.system(data_download)


def _refresh_data_folders():

    clean_data = f'rm -rf {str(DATA_ROOT_PATH)}'
    os.system(clean_data)

    create_data_dirs = f'mkdir {str(IMAGE_PATH)} -p; mkdir {str(LABEL_PATH)} -p'
    os.system(create_data_dirs)


def _unpack_data():

    zip_file_path = DATA_PATH / Path('train_images.zip')

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(IMAGE_PATH / Path('train'))

    zip_file_path = DATA_PATH / Path('test_images_phase1.zip')

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(IMAGE_PATH / Path('test'))


def data_prep():
    """
    Prepares the data in a way that YOLO likes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ai_crowd_api_key', type=str)
    args = parser.parse_args()

    _refresh_data_folders()
    _get_data(args.ai_crowd_api_key)
    _unpack_data()

    class_mapping = _get_class_mapping()
    data_frame = _read_training_data()
    _convert_yolo_format(data_frame, class_mapping)


if __name__ == "__main__":
    data_prep()
