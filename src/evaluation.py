from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from .config import CLASS_NAMES


def plot_class_distribution(labels: np.ndarray, title: str) -> None:
    counts = np.bincount(labels, minlength=26)

    plt.figure(figsize=(12, 4))
    plt.bar(CLASS_NAMES, counts)
    plt.title(title)
    plt.xlabel("Класс")
    plt.ylabel("Количество изображений")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_samples_per_class(images: np.ndarray, labels: np.ndarray, samples_per_class: int = 5) -> None:
    """Показывает по несколько примеров каждой буквы."""
    fig, axes = plt.subplots(26, samples_per_class, figsize=(samples_per_class * 1.6, 26 * 1.3))

    for class_idx in range(26):
        class_indices = np.where(labels == class_idx)[0]
        selected = np.random.choice(class_indices, size=samples_per_class, replace=False)

        for j, idx in enumerate(selected):
            ax = axes[class_idx, j]
            image = images[idx].reshape(28, 28)

            # EMNIST Letters хранится транспонированным — поворачиваем для корректного отображения.
            image = np.fliplr(np.rot90(image, k=3))

            ax.imshow(image, cmap="gray")
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(CLASS_NAMES[class_idx], rotation=0, labelpad=20, va="center", fontsize=9)

    plt.suptitle("Примеры изображений для каждой буквы", y=1.002)
    plt.tight_layout()
    plt.show()


def plot_history(histories: Dict[str, Dict[str, List[float]]], metric: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 5))

    for name, history in histories.items():
        if metric in history:
            plt.plot(history[metric], label=name)

    plt.title(title)
    plt.xlabel("Эпоха")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_learning_curves(name: str, history: Dict[str, List[float]]) -> None:
    """Рисует loss и accuracy (train + val) для одной модели на одном рисунке."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Кривые обучения — {name}", fontsize=13)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "train_accuracy" in history:
        axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Val", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_torch_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    confidences = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probabilities = torch.softmax(logits, dim=1)

            batch_conf, batch_pred = probabilities.max(dim=1)

            y_true.extend(y_batch.numpy())
            y_pred.extend(batch_pred.cpu().numpy())
            confidences.extend(batch_conf.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(confidences)


def get_tf_predictions(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = model.predict(X_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    return y_test, y_pred, confidences


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(26)))

    # Нормализуем по строкам для отображения процентов правильных ответов.
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(title, fontsize=13)

    for ax, data, fmt, subtitle in [
        (axes[0], cm, "d", "Абсолютные значения"),
        (axes[1], cm_norm, ".2f", "Нормализованные (по строкам)"),
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set_title(subtitle)
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_xticks(range(26))
        ax.set_yticks(range(26))
        ax.set_xticklabels(CLASS_NAMES, fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return cm


def get_top_confusions(cm: np.ndarray, top_n: int = 5) -> List[Tuple[str, str, int]]:
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    pairs = []
    for true_idx in range(cm_no_diag.shape[0]):
        for pred_idx in range(cm_no_diag.shape[1]):
            count = int(cm_no_diag[true_idx, pred_idx])
            if count > 0:
                pairs.append((CLASS_NAMES[true_idx], CLASS_NAMES[pred_idx], count))

    return sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]


def show_misclassified_examples(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    count: int = 15,
) -> None:
    error_indices = np.where(y_true != y_pred)[0]
    selected = error_indices[:count]

    cols = 5
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(selected):
        image = X_test[idx].reshape(28, 28)
        image = np.fliplr(np.rot90(image, k=3))

        axes[i].imshow(image, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(
            f"True: {CLASS_NAMES[y_true[idx]]}\n"
            f"Pred: {CLASS_NAMES[y_pred[idx]]}\n"
            f"Conf: {confidences[idx]:.2f}",
            fontsize=9,
        )

    for j in range(len(selected), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Примеры ошибок модели", fontsize=13)
    plt.tight_layout()
    plt.show()


def summarize_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, title: str = "") -> None:
    """Печатает sklearn classification report с названиями классов."""
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
    print(report)


def show_top_errors(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    top_n: int = 10,
) -> None:
    """Показывает топ-N неправильных предсказаний, отсортированных по уверенности модели.

    Высокая уверенность при ошибке — самый показательный тип ошибки (модель «уверена» в неправильном ответе).
    """
    error_mask = y_true != y_pred
    error_indices = np.where(error_mask)[0]

    # Сортируем по убыванию уверенности (самые «наглые» ошибки — сначала)
    sorted_by_conf = error_indices[np.argsort(confidences[error_indices])[::-1]]
    selected = sorted_by_conf[:top_n]

    cols = 5
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(selected):
        image = X_test[idx].reshape(28, 28)
        image = np.fliplr(np.rot90(image, k=3))

        axes[i].imshow(image, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(
            f"True: {CLASS_NAMES[y_true[idx]]}\n"
            f"Pred: {CLASS_NAMES[y_pred[idx]]}\n"
            f"Conf: {confidences[idx]:.2f}",
            fontsize=9,
        )

    for j in range(len(selected), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Топ-{top_n} ошибок (по убыванию уверенности)", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results: Dict[str, Dict], metric: str = "test_accuracy") -> None:
    """Столбчатая диаграмма сравнения моделей по выбранной метрике."""
    names = list(results.keys())
    values = [results[name][metric] for name in names]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(names)]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, values, color=colors, width=0.5)
    plt.ylim(min(values) - 0.05, 1.0)
    plt.ylabel("Accuracy")
    plt.title(f"Сравнение моделей — {metric}")
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()
