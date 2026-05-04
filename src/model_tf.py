import tensorflow as tf

from .config import INPUT_SIZE, LEARNING_RATE, NUM_CLASSES


def build_tf_mlp(dropout: bool = False, dropout_rate: float = 0.3) -> tf.keras.Model:
    """Создаёт MLP-модель на TensorFlow/Keras."""
    layers = [
        tf.keras.layers.Input(shape=(INPUT_SIZE,)),
        tf.keras.layers.Dense(512, activation="relu"),
    ]

    if dropout:
        layers.append(tf.keras.layers.Dropout(dropout_rate))

    layers.append(tf.keras.layers.Dense(256, activation="relu"))

    if dropout:
        layers.append(tf.keras.layers.Dropout(dropout_rate))

    layers.append(tf.keras.layers.Dense(128, activation="relu"))

    if dropout:
        layers.append(tf.keras.layers.Dropout(dropout_rate))

    layers.append(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def count_tf_parameters(model: tf.keras.Model) -> int:
    return model.count_params()
