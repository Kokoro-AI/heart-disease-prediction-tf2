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

def train(config):
    np.random.seed(2020)
    tf.random.set_seed(2020)

    # Useful data
    now = datetime.now()
    now_as_str = now.strftime('%Y_%m_%d-%H:%M:%S')

    # Output files
    checkpoint_path = f"{config['model.save_path']}"
    config_path = f"{config['output.config_path'].format(now_as_str)}"
    csv_output_path = f"{config['output.train_path'].format(now_as_str)}"
    tensorboard_summary_dir = f"{config['summary.save_path']}"
    summary_path = f"results/summary.csv"
    
    # Output dirs
    data_dir = f"data/"
    checkpoint_dir = checkpoint_path[:checkpoint_path.rfind('/')]
    config_dir = config_path[:config_path.rfind('/')]
    results_dir = csv_output_path[:csv_output_path.rfind('/')]

    # Create folder for model
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create output for train process
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file = open(f"{csv_output_path}", 'w') 
    file.write("")
    file.close()

    # Create folder for config
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # generate config file
    file = open(config_path, 'w')
    file.write(json.dumps(config, indent=2))
    file.close()

    # create summary file if not exists
    if not os.path.exists(summary_path):
        file = open(summary_path, 'w')
        file.write("datetime, model, config, min_loss, min_loss_accuracy\n")
        file.close()

    # Data loader
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ret = load(data_dir, config, ['train', 'val', 'test'])
    train_ds, train_features, _ = ret['train']
    val_ds, _, _ = ret['val']
    test_ds, _, _ = ret['test']

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_summary_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq='epoch'
    )

    logs_callback = tf.keras.callbacks.CSVLogger(
        csv_output_path,
        separator=',',
        append=False
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['train.patience']
    )

    time_start = time.time()
    # Compiles a model, prints the model summary, and saves the model diagram into a png file.
    input_shape = (len(train_features.keys()),)
    print(input_shape)
    model = create_model(input_shape=input_shape, learning_rate=config['train.lr'])
    
    model.summary()
    # tf.keras.utils.plot_model(model, "keras_model.png", show_shapes=True)

    # Trains the model.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['train.epochs'],
        use_multiprocessing=True,
        callbacks=[tensorboard_callback, logs_callback, model_checkpoint_callback, early_stop]
    )


    time_end = time.time()

    # Evaluates on test data.
    loss, acc = model.evaluate(val_ds)
    print("Evaluation finished!")

    summary = "{}, {}, df_model, {}, {}, {}\n".format(now_as_str, config['data.dataset'], config_path, loss, acc)
    print(summary)

    file = open(summary_path, 'a+') 
    file.write(summary)
    file.close()

    # Runs prediction on test data.
    predictions = tf.round(model.predict(test_ds)).numpy().flatten()
    print("Predictions on test data:")
    print(predictions)

    model_path = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=checkpoint_path)

    if not model_path:
        print("Skipping evaluation. No checkpoint found in: {}".format(checkpoint_dir))
    else:
        model_from_saved = tf.keras.models.load_model(model_path)
        model_from_saved.summary()

        # Runs test data through the reloaded model to make sure the results are same.
        predictions_from_saved = tf.round(model_from_saved.predict(test_ds)).numpy().flatten()
        np.testing.assert_array_equal(predictions_from_saved, predictions)

    elapsed = time_end - time_start
    h, min = elapsed//3600, elapsed%3600//60
    sec = elapsed-min*60

    print(f"Training took: {h:.2f}h {min:.2f}m {sec:.2f}s!")

def create_model(input_shape, learning_rate=0.01):
    """
        Constructs a model using various layers and compiles the model with proper
        optimizer/loss/metrics.
    """

    model = tf.keras.Sequential()

    for input_layer in get_feature_layer_inputs():
        model.add(input_layer)

    model.add(tf.keras.layers.Dense(15, kernel_initializer="normal", activation="relu", name="hidden_layer_1"))
    model.add(tf.keras.layers.Dense(64, kernel_initializer="normal", activation="relu", name="hidden_layer_2"))
    model.add(tf.keras.layers.Dropout(0.3, name="dropout_1"))
    model.add(tf.keras.layers.Dense(32, kernel_initializer="normal", activation="relu", name="hidden_layer_3"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="target"))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def get_feature_layer_inputs():
    """
        Builds feature layer inputs as keras Input.
    """

    feature_layer_inputs = []

    # numeric cols
    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
        feature_layer_inputs.append(tf.keras.Input(shape=(1,), name=header))

    feature_layer_inputs.append(tf.keras.Input(shape=(1,), name='thal', dtype=tf.string))
    feature_layer_inputs.append(tf.keras.Input(shape=(1,), name='sex', dtype=tf.string))
    feature_layer_inputs.append(tf.keras.Input(shape=(1,), name='cp', dtype=tf.string))
    feature_layer_inputs.append(tf.keras.Input(shape=(1,), name='slope', dtype=tf.string))

    return feature_layer_inputs