import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_heart(data_dir, config, splits):
    """
    Load heart dataset.
    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'
    Returns (dict): dictionary with keys as splits and values
    """

    RANDOM_SEED=42
    DATASET_NAME = "heart"
    DATASET_PATH = "/tf/data/heart.csv"

    data = pd.read_csv(DATASET_PATH)
    data = preprocess_data(data)

    train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    train, val = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED)

    # Defines datasets on the input data.
    batch_size = 32
    dfs = {'train': train, 'val': val, 'test': test} 

    ret = {}
    for split in splits:
        ds = df_to_dataset(dict(dfs[split]), shuffle=False, batch_size=batch_size)

        ret[split] = ds, dict(dfs[split])

    return ret

def preprocess_data(idata):
    data = idata.copy()

    data["thal"] = data["thal"].apply(str)
    data["sex"] = data["sex"].apply(str)
    data["cp"] = data["cp"].apply(str)
    data["slope"] = data["slope"].apply(str)

    return data

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds
