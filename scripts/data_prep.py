import pandas as pd
import os
import argparse
from pathlib import Path
import yaml

package_path = Path(__file__).resolve().parent.parent
data_path = package_path / Path('data')
label_path = data_path / Path('labels')

def convert_yolo_format(data_frame, class_mapping):

    for index, row in data_frame.iterrows():

        class_label = class_mapping[row['class_label']]
        x_center = float(row['bbx_xbr'] + row['bbx_xtl'])/(2*row['img_w'])
        y_center = float(row['bbx_ybr'] + row['bbx_ytl'])/(2*row['img_h'])
        width = float(row['bbx_xbr'] - row['bbx_xtl'])/row['img_w']
        height = float(row['bbx_ybr'] - row['bbx_ytl'])/row['img_h']

        with open(label_path / Path(row['img_fName'].split('.')[0] + '.txt'), 'w') as file:
            file.write(f'{class_label} {x_center} {y_center} {width} {height}\n')


def get_class_mapping(mapping_path):

    try:
        with open(mapping_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        classes = {value: key for key, value in yaml_data['names'].items()}
        return classes
    except FileNotFoundError:
        print("File not found.")

def read_csv(file_path):
    try:
        data_frame = pd.read_csv(file_path)
        return data_frame
    except FileNotFoundError:
        print("File not found.")


def format_data_folder():

    clean_data = f'rm -rf {str(data_path)}'
    os.system(clean_data)

    image_path = data_path / Path('images')
    create_data_dirs = f'mkdir {str(data_path)}; mkdir {str(image_path)}; mkdir {str(label_path)}'
    os.system(create_data_dirs)

def get_data(api_key):
    login = f'aicrowd login --api-key {api_key}'
    os.system(login)

    data_download = f'cd {str(data_path)}; aicrowd dataset download --challenge mosquitoalert-challenge-2023'
    os.system(data_download)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ai_crowd_api_key', type=str)
    args = parser.parse_args()

    format_data_folder()
    get_data(args.ai_crowd_api_key)

    mapping_file_path = package_path / Path('config/mosquito_alert.yaml')
    class_mapping = get_class_mapping(mapping_file_path)

    csv_file_path = data_path / Path('train.csv')
    data_frame = read_csv(csv_file_path)

    convert_yolo_format(data_frame, class_mapping)
