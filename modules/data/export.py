import numpy as np
import pandas as pd
import os
from os import path
import cv2
import zipfile
from PIL import Image
import toml

def image2dataset(compress=True):
    img_size = (28, 28)
    if compress:
        data_dict = {"inputs": [], "labels": []}
        with zipfile.ZipFile(CONFIG['path']['image_zip'], "r") as zipf:
            name_list = zipf.namelist()
            for f in name_list:
                if f.endswith('.png'):
                    with zipf.open(f) as fread:
                        buf = np.frombuffer(fread.read(), dtype=np.uint8)
                        np_buf = np.frombuffer(buf, np.uint8)
                        img_arr = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
                        image = Image.fromarray(img_arr, "L")
                        image.resize(img_size, Image.Resampling.LANCZOS)
                        data_dict['labels'].append(f.split('/')[-2])
                        data_dict['inputs'].append(np.array(image).flatten())

    input_arr = np.array(data_dict['inputs'])
    pd_data = {
        f"{(i // img_size[1]) + 1}x{(i%img_size[1]) + 1}": [] for i in range(img_size[0] * img_size[1])
    }
    for k_i, k in enumerate(pd_data.keys()):
        pd_data[k] = input_arr[:, k_i]
    df = pd.DataFrame(pd_data)
    df.insert(0, 'label', data_dict['labels'])
    df.to_csv(CONFIG['path']['dataset'], index=False)
