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
from tensorflow.keras import Input, Model
from src.datasets import load
from src.utils.callbacks import create_callbacks
from tensorflow.keras.layers import Dense, DenseFeatures, Dropout
from sklearn.model_selection import StratifiedKFold, train_test_split

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

    _, X, y = load(data_dir, config)

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

    for split in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40 + split)

        # Compiles a model, prints the model summary, and saves the model diagram into a png file.
        model = create_model(learning_rate=config['train.lr'])
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
                dict(X_train),
                y_train,
                validation_split=0.1,
                epochs=config['train.epochs'],
                batch_size=config['data.batch_size'],
                use_multiprocessing=True,
                callbacks=callbacks
            )

        # evaluate the model
        scores = model.evaluate(dict(X_test), y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

        # Runs prediction on test data.
        predictions = tf.round(model.predict(dict(X_test))).numpy().flatten()
        print("Predictions on test data:")
        print(predictions)

        model_path = tf.train.latest_checkpoint(split_checkpoint_dir, latest_filename=split_checkpoint_path)

        if not model_path:
            print("Skipping evaluation. No checkpoint found in: {}".format(split_checkpoint_dir))
        else:
            model_from_saved = tf.keras.models.load_model(model_path)
            model_from_saved.summary()

            # Runs test data through the reloaded model to make sure the results are same.
            predictions_from_saved = tf.round(model_from_saved.predict(dict(X_test))).numpy().flatten()
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

def create_model(learning_rate=0.01):
    """
        Constructs a model using various layers and compiles the model with proper
        optimizer/loss/metrics.
    """

    feature_columns, feature_layer_inputs = get_feature_transform()
    feature_layer = DenseFeatures(feature_columns, name="feature")
    feature_layer_outputs = feature_layer(feature_layer_inputs)

    x = Dense(128, kernel_initializer="normal", activation="relu", name="hidden_layer_1")(feature_layer_outputs)
    x = Dropout(0.2, name="dropout_1")(x)
    x = Dense(128, kernel_initializer="normal", activation="relu", name="hidden_layer_2")(x)
    baggage_pred = Dense(1, activation="sigmoid", name="target")(x)
    
    model = Model(inputs=[v for v in feature_layer_inputs.values()], outputs=baggage_pred)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def get_feature_transform():
    """
        Builds a DenseFeatures layer as feature transformation.

        The function handles all feature transformation such as bucketizing,
        vectorizing (one-hot encoding), etc.
    """

    feature_columns = []
    feature_layer_inputs = {}

    # numeric cols
    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
        feature_columns.append(tf.feature_column.numeric_column(header))
        feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name=header)

    # bucketized cols
    age = tf.feature_column.numeric_column("age")
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    # indicator cols
    thal = tf.feature_column.categorical_column_with_vocabulary_list(
        'thal', ['fixed', 'normal', 'reversible'])
    thal_one_hot = tf.feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)
    feature_layer_inputs['thal'] = tf.keras.Input(shape=(1,), name='thal', dtype=tf.string)

    sex = tf.feature_column.categorical_column_with_vocabulary_list(
        'sex', ['0', '1'])
    sex_one_hot = tf.feature_column.indicator_column(sex)
    feature_columns.append(sex_one_hot)
    feature_layer_inputs['sex'] = tf.keras.Input(shape=(1,), name='sex', dtype=tf.string)

    cp = tf.feature_column.categorical_column_with_vocabulary_list(
        'cp', ['0', '1', '2', '3'])
    cp_one_hot = tf.feature_column.indicator_column(cp)
    feature_columns.append(cp_one_hot)
    feature_layer_inputs['cp'] = tf.keras.Input(shape=(1,), name='cp', dtype=tf.string)

    slope = tf.feature_column.categorical_column_with_vocabulary_list(
        'slope', ['0', '1', '2'])
    slope_one_hot = tf.feature_column.indicator_column(slope)
    feature_columns.append(slope_one_hot)
    feature_layer_inputs['slope'] = tf.keras.Input(shape=(1,), name='slope', dtype=tf.string)

    return feature_columns, feature_layer_inputs