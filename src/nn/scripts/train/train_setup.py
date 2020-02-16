"""
Logic for model creation, training launching and actions needed to be
accomplished during training (metrics monitor, model saving etc.)
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from src.nn.datasets import load

def train(config):
    np.random.seed(2020)
    tf.random.set_seed(2020)

    # Useful data
    now = datetime.now()
    now_as_str = now.strftime('%Y_%m_%d-%H:%M:%S')

    # Output files
    model_file = f"{config['model.save_path']}"
    config_file = f"{config['output.config_path'].format(now_as_str)}"
    csv_output_file = f"{config['output.train_path'].format(now_as_str)}"
    summary_dir = f"{config['summary.save_path']}"
    
    # Output dirs
    data_dir = f"data/"
    model_dir = model_file[:model_file.rfind('/')]
    config_dir = config_file[:config_file.rfind('/')]
    results_dir = csv_output_file[:csv_output_file.rfind('/')]

    # Create folder for model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create output for train process
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file = open(f"{csv_output_file}", 'w') 
    file.write("")
    file.close()

    # Create folder for config
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # generate config file
    file = open(config_file, 'w')
    file.write(json.dumps(config, indent=2))
    file.close()

    # Data loader
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ret, feature_columns = load(data_dir, config, ['train', 'val'])
    train_ds = ret['train']
    val_ds = ret['val']

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=summary_dir, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    logs_callback = tf.keras.callbacks.CSVLogger(csv_output_file, separator=',', append=False)
    mc_callback = tf.keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=2, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['train.epochs'],
        use_multiprocessing=True,
        callbacks=[tb_callback, logs_callback, mc_callback]
    )

