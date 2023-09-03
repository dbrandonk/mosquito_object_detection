"""A very brittle script that setups up the data (https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023) nicely to use with YOLO."""  # pylint: disable=line-too-long

from pathlib import Path
import argparse
import shutil
import zipfile

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import yaml

PACKAGE_PATH = Path(__file__).resolve().parent.parent
DATA_ROOT_PATH = PACKAGE_PATH / Path('data')
DATA_PATH = DATA_ROOT_PATH / Path('mosquito_alert_2023')
IMAGE_PATH = DATA_PATH / Path('images')
GRAB_IMAGE_PATH = IMAGE_PATH / Path('final')
TRAIN_IMAGE_PATH = IMAGE_PATH / Path('train')
TRAIN_LABEL_PATH = DATA_PATH / Path('labels/train')
VAL_IMAGE_PATH = IMAGE_PATH / Path('val')
VAL_LABEL_PATH = DATA_PATH / Path('labels/val')
TRAIN_CLASSIFICATION_PATH = DATA_PATH / Path('classification/train')
VAL_CLASSIFICATION_PATH = DATA_PATH / Path('classification/val')
CLASS_MAPPING_PATH = PACKAGE_PATH / Path('config/mosquito_alert.yaml')
TRAINING_DATA_PATH = DATA_PATH / Path('phase2_train_v0.csv')

def _clean_up():

    # cleaning up
    shutil.rmtree(GRAB_IMAGE_PATH)

def _convert_yolo_format(data_frame, class_mapping, balance_data):  # pylint: disable=too-many-locals

    x_train, x_test = train_test_split(data_frame, test_size=0.2, shuffle=True, random_state=42)

    if balance_data:
        target = x_train['class_label']
        oversampler = RandomOverSampler(random_state=42)
        x_train, _ = oversampler.fit_resample(x_train, target)

    for dataset_idx, dataset in enumerate([x_train, x_test]):
        for idx, row in dataset.iterrows():

            if dataset_idx == 0:
                image_placement_path = TRAIN_IMAGE_PATH
                label_placement_path = TRAIN_LABEL_PATH
            else:
                image_placement_path = VAL_IMAGE_PATH
                label_placement_path = VAL_LABEL_PATH

            shutil.copy(
                GRAB_IMAGE_PATH /
                Path(
                    row['img_fName']),
                image_placement_path /
                Path(f'img{idx}.jpeg'))

            class_label = class_mapping[row['class_label']]
            x_center = float(row['bbx_xbr'] + row['bbx_xtl']) / (2 * row['img_w'])
            y_center = float(row['bbx_ybr'] + row['bbx_ytl']) / (2 * row['img_h'])
            width = float(row['bbx_xbr'] - row['bbx_xtl']) / row['img_w']
            height = float(row['bbx_ybr'] - row['bbx_ytl']) / row['img_h']

            with open(label_placement_path / Path(f'img{idx}.txt'),
                      'w', encoding='utf-8') as file:
                file.write(f'{class_label} {x_center} {y_center} {width} {height}\n')

def _org_classification_data(data_frame, class_mapping, balance_data):  # pylint: disable=too-many-locals

    # lets make all of the sub folders.
    for cls in class_mapping:

        class_label = cls.replace('/', '_') # just in case, otherwise stucture will break.

        # remove any data that might be there.
        try:
            shutil.rmtree(TRAIN_CLASSIFICATION_PATH/ Path(class_label))
            shutil.rmtree(VAL_CLASSIFICATION_PATH/ Path(class_label))
        except FileNotFoundError:
            pass

        shutil.os.makedirs(TRAIN_CLASSIFICATION_PATH/ Path(class_label))
        shutil.os.makedirs(VAL_CLASSIFICATION_PATH/ Path(class_label))

    x_train, x_test = train_test_split(data_frame, test_size=0.2, shuffle=True, random_state=42)

    if balance_data:
        target = x_train['class_label']
        oversampler = RandomOverSampler(random_state=42)
        x_train, _ = oversampler.fit_resample(x_train, target)

    for dataset_idx, dataset in enumerate([x_train, x_test]):
        for idx, row in dataset.iterrows():

            if dataset_idx == 0:
                image_placement_path = TRAIN_CLASSIFICATION_PATH
            else:
                image_placement_path = VAL_CLASSIFICATION_PATH

            class_label = row['class_label'].replace('/', '_') # just in case, otherwise stucture will break.
            shutil.copy(GRAB_IMAGE_PATH / Path( row['img_fName']), image_placement_path / Path(class_label + f'/img{idx}.jpeg'))


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
    shutil.os.system(login)

    ai_crowd_cmd = 'aicrowd dataset download --challenge mosquitoalert-challenge-2023'
    data_download = f'cd {str(DATA_PATH)}; {ai_crowd_cmd}'
    shutil.os.system(data_download)


def _refresh_data_folders():

    # remove any data that might be there.
    try:
        shutil.rmtree(DATA_ROOT_PATH)
    except FileNotFoundError:
        pass

    shutil.os.makedirs(TRAIN_CLASSIFICATION_PATH)
    shutil.os.makedirs(TRAIN_IMAGE_PATH)
    shutil.os.makedirs(TRAIN_LABEL_PATH)
    shutil.os.makedirs(VAL_CLASSIFICATION_PATH)
    shutil.os.makedirs(VAL_IMAGE_PATH)
    shutil.os.makedirs(VAL_LABEL_PATH)


def _unpack_data():

    zip_file_path = DATA_PATH / Path('phase2_train_v0.zip')

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(IMAGE_PATH)

def data_prep():
    """
    Prepares the data in a way that YOLO likes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ai_crowd_api_key', required=True, type=str)
    parser.add_argument('--balance_data', action='store_true')
    args = parser.parse_args()

    _refresh_data_folders()
    _get_data(args.ai_crowd_api_key)
    _unpack_data()

    class_mapping = _get_class_mapping()
    data_frame = _read_training_data()
    _convert_yolo_format(data_frame, class_mapping, args.balance_data)
    _org_classification_data(data_frame, class_mapping, args.balance_data)

    _clean_up()


if __name__ == "__main__":
    data_prep()
