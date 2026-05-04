from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import tensorflow as tf
from PIL import Image

from .config import CLASS_NAMES


def preprocess_custom_image(image_path: str) -> np.ndarray:
    """Готовит пользовательское изображение буквы к инференсу."""
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))

    arr = np.array(image).astype("float32")

    # Если фон светлый, а буква тёмная — инвертируем.
    if arr.mean() > 127:
        arr = 255 - arr

    arr = arr / 255.0
    arr = arr.reshape(1, 784)

    return arr


def predict_letter(image_path: str, model, framework: str) -> Tuple[str, float]:
    """Возвращает предсказанную букву и уверенность.

    framework:
    - "torch"
    - "tensorflow" или "tf"
    """
    x = preprocess_custom_image(image_path)

    if framework.lower() == "torch":
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32).to(device)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        return CLASS_NAMES[int(predicted_idx.item())], float(confidence.item())

    if framework.lower() in {"tensorflow", "tf"}:
        probabilities = model.predict(x, verbose=0)
        predicted_idx = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(np.max(probabilities))

        return CLASS_NAMES[predicted_idx], confidence

    raise ValueError("framework должен быть 'torch', 'tensorflow' или 'tf'")
