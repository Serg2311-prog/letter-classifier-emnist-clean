import random
import numpy as np
import torch
import tensorflow as tf

SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 26
INPUT_SIZE = 784
IMAGE_SIZE = 28
CLASS_NAMES = [chr(ord("A") + i) for i in range(NUM_CLASSES)]


def set_seed(seed: int = SEED) -> None:
    """Фиксирует random seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
