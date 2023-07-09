import os
from pathlib import Path

package_path = Path(__file__).resolve().parent.parent
data_path = package_path / Path('data')

clean_data = f'rm -rf {str(data_path)}'
os.system(clean_data)

image_path = data_path / Path('images')
label_path = data_path / Path('labels')
create_data_dirs = f'mkdir {str(data_path)}; mkdir {str(image_path)}; mkdir {str(label_path)}'
os.system(create_data_dirs)

data_download = f'cd {str(data_path)}; aicrowd dataset download --challenge mosquitoalert-challenge-2023'
os.system(data_download)
