from pathlib import Path
import argparse
import os
import zipfile

import pandas as pd
import yaml

package_path = Path(__file__).resolve().parent.parent
data_parent_path = package_path / Path('data')
data_path = data_parent_path / Path('mosquito_alert_2023')
image_path = data_path / Path('images')
label_path = data_path / Path('labels/train')


def convert_yolo_format(data_frame, class_mapping):

    for _, row in data_frame.iterrows():

        class_label = class_mapping[row['class_label']]
        x_center = float(row['bbx_xbr'] + row['bbx_xtl']) / (2 * row['img_w'])
        y_center = float(row['bbx_ybr'] + row['bbx_ytl']) / (2 * row['img_h'])
        width = float(row['bbx_xbr'] - row['bbx_xtl']) / row['img_w']
        height = float(row['bbx_ybr'] - row['bbx_ytl']) / row['img_h']

        with open(label_path / Path(row['img_fName'].split('.')[0] + '.txt'),
                'w', encoding='utf-8') as file:
            file.write(f'{class_label} {x_center} {y_center} {width} {height}\n')


def get_class_mapping(mapping_path):

    try:
        with open(mapping_path, 'rb') as file:
            yaml_data = yaml.safe_load(file)
        classes = {value: key for key, value in yaml_data['names'].items()}
    except FileNotFoundError:
        print("File not found.")

    return classes


def read_csv(file_path):
    try:
        data_frame = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found.")

    return data_frame


def format_data_folder():

    clean_data = f'rm -rf {str(data_parent_path)}'
    os.system(clean_data)

    create_data_dirs = f'mkdir {str(image_path)} -p; mkdir {str(label_path)} -p'
    os.system(create_data_dirs)


def get_data(api_key):
    login = f'aicrowd login --api-key {api_key}'
    os.system(login)

    ai_crowd_cmd = 'aicrowd dataset download --challenge mosquitoalert-challenge-2023'
    data_download = f'cd {str(data_path)}; {ai_crowd_cmd}'
    os.system(data_download)


def unpack_data():

    zip_file_path = data_path / Path('train_images.zip')

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(image_path / Path('train'))

    zip_file_path = data_path / Path('test_images_phase1.zip')

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(image_path / Path('test'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ai_crowd_api_key', type=str)
    args = parser.parse_args()

    format_data_folder()
    get_data(args.ai_crowd_api_key)
    unpack_data()

    mapping_file_path = package_path / Path('config/mosquito_alert.yaml')
    class_mapping = get_class_mapping(mapping_file_path)

    csv_file_path = data_path / Path('train.csv')
    data_frame = read_csv(csv_file_path)

    convert_yolo_format(data_frame, class_mapping)


if __name__ == "__main__":
    main()
