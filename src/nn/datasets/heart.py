import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split

def load_heart(data_dir, config, splits):
    """
    Load heart dataset.
    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'
    Returns (dict): dictionary with keys as splits and values as tf.Dataset
    """

    RANDOM_SEED=42
    DATASET_NAME = "heart"
    DATASET_PATH = "/tf/data/heart.csv"

    data = pd.read_csv(DATASET_PATH)

    data, feature_columns = preprocess_data(data)

    train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)

    ret = {}
    for split in splits:
        if split in ['val', 'test']:
            split_data = test
        else:
            split_data = train

        ret[split] = create_dataset(split_data)

    return ret, feature_columns

def create_dataset(dataframe, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    return Dataset.from_tensor_slices((dict(dataframe), labels)) \
                  .shuffle(buffer_size=len(dataframe)) \
                  .batch(batch_size)

def preprocess_data(input_data):
    """
        Feature columns allow you to bridge/process the raw data in your dataset to fit
        your model input data requirements. Furthermore, you can separate the model building process
        from the data preprocessing.

        Apart from the numerical features, we’re putting patient age into discrete ranges (buckets).
        Furthermore, thal, sex, cp, and slope are categorical and we map them to such.
    """

    data = input_data.copy()

    X = data.loc[:, data.columns != 'target']
    y = data.iloc[:, -1]

    feature_columns = []

    # numeric cols
    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
        feature_columns.append(tf.feature_column.numeric_column(header))

    # bucketized cols
    age = tf.feature_column.numeric_column("age")
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    # indicator cols
    data["thal"] = data["thal"].apply(str)
    thal = tf.feature_column.categorical_column_with_vocabulary_list(
        'thal', ['3', '6', '7'])
    thal_one_hot = tf.feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)

    data["sex"] = data["sex"].apply(str)
    sex = tf.feature_column.categorical_column_with_vocabulary_list(
        'sex', ['0', '1'])
    sex_one_hot = tf.feature_column.indicator_column(sex)
    feature_columns.append(sex_one_hot)

    data["cp"] = data["cp"].apply(str)
    cp = tf.feature_column.categorical_column_with_vocabulary_list(
        'cp', ['0', '1', '2', '3'])
    cp_one_hot = tf.feature_column.indicator_column(cp)
    feature_columns.append(cp_one_hot)

    data["slope"] = data["slope"].apply(str)
    slope = tf.feature_column.categorical_column_with_vocabulary_list(
        'slope', ['0', '1', '2'])
    slope_one_hot = tf.feature_column.indicator_column(slope)
    feature_columns.append(slope_one_hot)


    # embedding cols
    thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
    feature_columns.append(thal_embedding)

    # crossed cols
    age_thal_crossed = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
    age_thal_crossed = tf.feature_column.indicator_column(age_thal_crossed)
    feature_columns.append(age_thal_crossed)

    cp_slope_crossed = tf.feature_column.crossed_column([cp, slope], hash_bucket_size=1000)
    cp_slope_crossed = tf.feature_column.indicator_column(cp_slope_crossed)
    feature_columns.append(cp_slope_crossed)

    return data, feature_columns
