import pandas as pd
import argparse
from pathlib import Path
import yaml

def convert_yolo_format(data_frame, class_mapping, output_path):

    for index, row in data_frame.iterrows():

        class_label = class_mapping[row['class_label']]
        x_center = float(row['bbx_xbr'] + row['bbx_xtl'])/(2*row['img_w'])
        y_center = float(row['bbx_ybr'] + row['bbx_ytl'])/(2*row['img_h'])
        width = float(row['bbx_xbr'] - row['bbx_xtl'])/row['img_w']
        height = float(row['bbx_ybr'] - row['bbx_ytl'])/row['img_h']

        with open(output_path / Path(row['img_fName'].split('.')[0] + '.txt'), 'w') as file:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=Path)
    parser.add_argument('--mapping_path', type=Path)
    parser.add_argument('--output_path', type=Path)
    args = parser.parse_args()

    data_frame = read_csv(args.label_path)
    class_mapping = get_class_mapping(args.mapping_path)

    convert_yolo_format(data_frame, class_mapping, args.output_path)

