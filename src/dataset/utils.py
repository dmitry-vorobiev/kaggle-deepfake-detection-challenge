import os

import numpy as np
import pandas as pd


def read_labels(base_path):
    if not os.path.isdir(base_path):
        raise ValueError('Invalid data dir')
    chunk_dirs = os.listdir(base_path)
    labels = []
    for dir_name in chunk_dirs:
        path = os.path.join(base_path, dir_name, 'metadata.json')
        df = pd.read_json(path).T
        df['dir'] = dir_name
        df['label'] = (df['label'] == 'FAKE').astype(np.uint8)
        df.drop(['split'], axis=1, inplace=True)
        labels.append(df)
    return pd.concat(labels)