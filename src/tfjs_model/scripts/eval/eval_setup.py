"""
Logic for evaluation procedure of saved model.
"""

import tensorflow as tf
import tensorflowjs as tfjs
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, accuracy_score

from src.datasets import load

def eval(config):
    # Files path
    model_file = f"{config['model.path']}"
    data_dir = f"data/"

    ret = load(data_dir, config, ['test'], numeric=True)
    _, X_test, y_test = ret['test']

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    model = tf.keras.models.load_model(model_file)
    model.summary()

    if config['model.weights_save_path'] != "":
        model.save_weights(f"{config['model.weights_save_path']}")

    if config['model.json_save_path'] != "":
        tfjs.converters.save_keras_model(model, f"{config['model.json_save_path']}")

    predictions = tf.round(model.predict(X_test)).numpy().flatten()
    print('Results for Binary Model')
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
