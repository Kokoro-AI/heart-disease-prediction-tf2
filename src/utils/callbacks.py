import tensorflow as tf

def create_callbacks(tensorboard_summary_dir, csv_output_path, checkpoint_path, patience):
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
        patience=patience
    )

    return [tensorboard_callback, logs_callback, model_checkpoint_callback, early_stop]