"""
Logic for evaluation procedure of saved model.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix

from src.nn.datasets import load

def eval(config):
    # Files path
    model_file = f"{config['model.save_path']}"
    data_dir = f"data/"

    ret = load(data_dir, config, ['test'])
    _, test_features = ret['test']

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    model = tf.keras.models.load_model(model_file)

    model.summary()

    predictions = tf.round(model.predict({"feature": test_features})).numpy().flatten()
    print(predictions)
