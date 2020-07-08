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
from src.datasets import load
from tensorflow.keras import Input, Model
from src.utils.callbacks import create_callbacks
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold

def train(config):
    np.random.seed(2020)
    tf.random.set_seed(2020)

    # Useful data
    now = datetime.now()
    now_as_str = now.strftime('%y_%m_%d-%H:%M:%S')

    # Output files
    checkpoint_path = config['model.save_path']
    config_path = config['output.config_path'].format(date=now_as_str)
    csv_output_path = config['output.train_path'].format(date=now_as_str)
    tensorboard_summary_dir = config['summary.save_path']
    summary_path = "results/summary.csv"

    # Output dirs
    data_dir = "data/"
    config_dir = config_path[:config_path.rfind('/')]
    output_dir = csv_output_path[:csv_output_path.rfind('/')]

    # Create folder for config
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # generate config file
    file = open(config_path, 'w')
    file.write(json.dumps(config, indent=2))
    file.close()
    
    file = open(csv_output_path, 'w') 
    file.write("")
    file.close()

    # create summary file if not exists
    if not os.path.exists(summary_path):
        file = open(summary_path, 'w')
        file.write("datetime, model, config, acc_std, acc_mean\n")
        file.close()

    # Data loader
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    _, X, y = load(data_dir, config, use_feature_transform=True)

    # Defines datasets on the input data.
    batch_size = config['data.batch_size']

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    time_start = time.time()

    # define 10-fold cross validation test harness
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cvscores = []
    print ("Running model performance validation... please wait!")

    for split, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Compiles a model, prints the model summary, and saves the model diagram into a png file.
        input_shape = (X_train.shape[1],)
        model = create_model(input_shape=input_shape, learning_rate=config['train.lr'])
        model.summary()

        split_checkpoint_path = checkpoint_path.format(split=split)
        split_results_path = csv_output_path.format(split=split)

        split_checkpoint_dir = split_checkpoint_path[:split_checkpoint_path.rfind('/')]
        split_results_dir = split_results_path[:split_results_path.rfind('/')]

        # Create folder for model
        if not os.path.exists(split_checkpoint_dir):
            os.makedirs(split_checkpoint_dir)

        # Create output for train process
        if not os.path.exists(split_results_dir):
            os.makedirs(split_results_dir)
        
        tf.keras.utils.plot_model(model, os.path.join(split_results_dir, "keras_model.png"), show_shapes=True, show_layer_names=False)

        callbacks = create_callbacks(
            tensorboard_summary_dir.format(split=split),
            split_results_path,
            split_checkpoint_path,
            patience=config['train.patience']
        )

        # Fit the model
        with tf.device(device_name):
            history = model.fit(
                X_train,
                y_train,
                validation_split=0.1,
                epochs=config['train.epochs'],
                batch_size=config['data.batch_size'],
                use_multiprocessing=True,
                callbacks=callbacks
            )

        # evaluate the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

        # Runs prediction on test data.
        predictions = tf.round(model.predict(X_test)).numpy().flatten()
        print("Predictions on test data:")
        print(predictions)

        model_path = tf.train.latest_checkpoint(split_checkpoint_dir, latest_filename=split_checkpoint_path)

        if not model_path:
            print("Skipping evaluation. No checkpoint found in: {}".format(split_checkpoint_dir))
        else:
            model_from_saved = tf.keras.models.load_model(model_path)
            model_from_saved.summary()

            # Runs test data through the reloaded model to make sure the results are same.
            predictions_from_saved = tf.round(model_from_saved.predict(X_test)).numpy().flatten()
            np.testing.assert_array_equal(predictions_from_saved, predictions)

    print ("Done.")
    print ("Summary report on mean and std.")
    # The average and standard deviation of the model performance 
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    time_end = time.time()

    summary = "{}, {}, df, {}, {}, {}\n".format(now_as_str, config['data.dataset'], config_path, np.std(cvscores), np.mean(cvscores))
    print(summary)

    file = open(summary_path, 'a+') 
    file.write(summary)
    file.close()

    elapsed = time_end - time_start
    h, min = elapsed//3600, elapsed%3600//60
    sec = elapsed-min*60

    print(f"Training took: {h:.2f}h {min:.2f}m {sec:.2f}s!")

def create_model(input_shape, learning_rate=0.01):
    """
        Constructs a model using various layers and compiles the model with proper
        optimizer/loss/metrics.
    """

    inputs = Input(shape=input_shape, name="feature")
    x = Dense(128, kernel_initializer="normal", activation="relu", name="hidden_layer_1")(inputs)
    x = Dropout(0.2, name="dropout_1")(x)
    x = Dense(128, kernel_initializer="normal", activation="relu", name="hidden_layer_2")(x)
    baggage_pred = Dense(1, activation="sigmoid", name="target")(x)

    model = Model(inputs=inputs, outputs=baggage_pred, name="hdprediction")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model
