import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_heart(data_dir, config, splits, use_feature_transform=False):
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

    train, test = train_test_split(data, test_size=0.1, random_state=RANDOM_SEED)
    train, val = train_test_split(train, test_size=0.1, random_state=RANDOM_SEED)

    # Defines datasets on the input data.
    batch_size = config['data.batch_size']
    feature_transform = get_feature_transform()

    dfs = {'train': train, 'val': val, 'test': test} 

    ret = {}
    for split in splits:
        if use_feature_transform == True:
            features = feature_transform(dict(dfs[split])).numpy()
            ds = df_to_dataset(features, dfs[split]["target"].values, shuffle=False, batch_size=batch_size)
            ret[split] = ds, features
        else:
            labels = dfs[split].pop('target')
            ds = df_to_dataset(dict(dfs[split]), labels, shuffle=False, batch_size=batch_size)
            ret[split] = ds, dfs[split]

    return ret

def preprocess_data(idata):
    data = idata.copy()

    data["thal"] = data["thal"].apply(str)
    data["sex"] = data["sex"].apply(str)
    data["cp"] = data["cp"].apply(str)
    data["slope"] = data["slope"].apply(str)

    return data

def df_to_dataset(features, labels, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))
    ds = ds.batch(batch_size)
    return ds

def get_feature_transform():
    """
        Builds a DenseFeatures layer as feature transformation.

        The function handles all feature transformation such as bucketizing,
        vectorizing (one-hot encoding), etc.
    """

    feature_columns = []

    # numeric cols
    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
        feature_columns.append(tf.feature_column.numeric_column(header))

    # bucketized cols
    age = tf.feature_column.numeric_column("age")
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    # indicator cols
    thal = tf.feature_column.categorical_column_with_vocabulary_list(
        'thal', ['3', '6', '7'])
    thal_one_hot = tf.feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)

    sex = tf.feature_column.categorical_column_with_vocabulary_list(
        'sex', ['0', '1'])
    sex_one_hot = tf.feature_column.indicator_column(sex)
    feature_columns.append(sex_one_hot)

    cp = tf.feature_column.categorical_column_with_vocabulary_list(
        'cp', ['0', '1', '2', '3'])
    cp_one_hot = tf.feature_column.indicator_column(cp)
    feature_columns.append(cp_one_hot)

    slope = tf.feature_column.categorical_column_with_vocabulary_list(
        'slope', ['0', '1', '2'])
    slope_one_hot = tf.feature_column.indicator_column(slope)
    feature_columns.append(slope_one_hot)

    return tf.keras.layers.DenseFeatures(feature_columns)
