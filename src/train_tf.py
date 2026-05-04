from __future__ import annotations

import time
from typing import Dict, List, Tuple

import tensorflow as tf

from .config import EPOCHS, LEARNING_RATE


class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self.start_time)


def train_tf_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = EPOCHS,
) -> Tuple[tf.keras.Model, Dict[str, List[float]]]:
    """Обучает TensorFlow модель и возвращает историю метрик."""
    timer = EpochTimer()

    history_obj = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[timer],
        verbose=1,
    )

    history = {
        "train_loss": history_obj.history["loss"],
        "train_accuracy": history_obj.history["accuracy"],
        "val_loss": history_obj.history["val_loss"],
        "val_accuracy": history_obj.history["val_accuracy"],
        "epoch_time": timer.epoch_times,
    }

    return model, history
