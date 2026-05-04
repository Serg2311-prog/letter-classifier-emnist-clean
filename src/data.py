from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
import tensorflow as tf

from .config import BATCH_SIZE, SEED


class EMNISTLabelOffsetDataset(Dataset):
    """Обёртка: переводит метки EMNIST Letters из диапазона 1–26 в 0–25."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, int(label) - 1


@dataclass
class TorchData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    full_train_raw: Dataset
    test_raw: Dataset


@dataclass
class NumpyData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class TFData:
    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset
    test_dataset: tf.data.Dataset


def get_transform() -> transforms.Compose:
    """Возвращает трансформацию для MLP: изображение разворачивается в вектор 784."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])


def get_transform_2d() -> transforms.Compose:
    """Возвращает трансформацию для CNN: изображение остаётся тензором (1, 28, 28)."""
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def load_raw_emnist(root: str = "./data") -> Tuple[Dataset, Dataset]:
    """Загружает EMNIST Letters через torchvision (MLP: изображение → вектор 784)."""
    transform = get_transform()

    full_train = datasets.EMNIST(
        root=root,
        split="letters",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.EMNIST(
        root=root,
        split="letters",
        train=False,
        download=True,
        transform=transform,
    )

    return full_train, test_dataset


def load_raw_emnist_2d(root: str = "./data") -> Tuple[Dataset, Dataset]:
    """Загружает EMNIST Letters через torchvision (CNN: изображение → тензор (1, 28, 28))."""
    transform = get_transform_2d()

    full_train = datasets.EMNIST(
        root=root,
        split="letters",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.EMNIST(
        root=root,
        split="letters",
        train=False,
        download=True,
        transform=transform,
    )

    return full_train, test_dataset


def create_torch_loaders(
    full_train_raw: Dataset,
    test_raw: Dataset,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = 0.1,
    seed: int = SEED,
) -> TorchData:
    """Создаёт PyTorch DataLoader с train/validation split 90/10."""
    full_train = EMNISTLabelOffsetDataset(full_train_raw)
    test_dataset = EMNISTLabelOffsetDataset(test_raw)

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return TorchData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        full_train_raw=full_train_raw,
        test_raw=test_raw,
    )


def create_cnn_loaders(
    full_train_raw: Dataset,
    test_raw: Dataset,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = 0.1,
    seed: int = SEED,
) -> TorchData:
    """Создаёт DataLoader-ы для CNN (изображения формата (1, 28, 28))."""
    full_train = EMNISTLabelOffsetDataset(full_train_raw)
    test_dataset = EMNISTLabelOffsetDataset(test_raw)

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return TorchData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        full_train_raw=full_train_raw,
        test_raw=test_raw,
    )


def raw_emnist_to_numpy(
    full_train_raw: Dataset,
    test_raw: Dataset,
    val_ratio: float = 0.1,
    seed: int = SEED,
) -> NumpyData:
    """Конвертирует torchvision EMNIST в NumPy для TensorFlow."""
    X_full = full_train_raw.data.numpy().reshape(-1, 784).astype("float32") / 255.0
    y_full = full_train_raw.targets.numpy().astype("int32") - 1

    X_test = test_raw.data.numpy().reshape(-1, 784).astype("float32") / 255.0
    y_test = test_raw.targets.numpy().astype("int32") - 1

    rng = np.random.default_rng(seed)
    indices = np.arange(len(X_full))
    rng.shuffle(indices)

    val_size = int(len(X_full) * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return NumpyData(
        X_train=X_full[train_idx],
        y_train=y_full[train_idx],
        X_val=X_full[val_idx],
        y_val=y_full[val_idx],
        X_test=X_test,
        y_test=y_test,
    )


def create_tf_datasets(np_data: NumpyData, batch_size: int = BATCH_SIZE) -> TFData:
    """Создаёт tf.data.Dataset для train/validation/test."""
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((np_data.X_train, np_data.y_train))
        .shuffle(1000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((np_data.X_val, np_data.y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((np_data.X_test, np_data.y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return TFData(train_dataset, val_dataset, test_dataset)
