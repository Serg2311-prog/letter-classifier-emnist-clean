# Letter Classifier — EMNIST Letters

**Автор:** Сергей Смирнов  
**Датасет:** [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset) — рукописные буквы A–Z (26 классов, 145 600 изображений 28×28)

---

## Описание проекта

Полный цикл машинного обучения для задачи классификации рукописных букв английского алфавита.  
Реализованы и сравнены три архитектуры:

| Модель | Архитектура | Framework | Параметры |
|--------|-------------|-----------|-----------|
| **MLP baseline** | 784→512→256→128→26 | PyTorch / TensorFlow | ~500K |
| **MLP + Dropout** | 784→512→256→128→26 + Dropout(0.3) | PyTorch / TensorFlow | ~500K |
| **CNN** | Conv(32)→Conv(64)→BN→MaxPool→FC(256)→26 | PyTorch | ~200K |

---

## Результаты (Test Set)

| Model                      | Test Accuracy |
|---------------------------|--------------|
| MLP baseline (PyTorch)    | 91.48%       |
| MLP + Dropout (PyTorch)   | 91.31%       |
| MLP baseline (TensorFlow) | 90.68%       |
| MLP + Dropout (TensorFlow)| 91.10%       |
| **CNN (PyTorch)**         | **94.21%**   |

CNN показывает значительно лучший результат за счёт извлечения пространственных признаков.

---

## Архитектуры моделей

### MLP (Multi-Layer Perceptron)

```
Input (784) → Linear(512) → ReLU → [Dropout(0.3)]
           → Linear(256) → ReLU → [Dropout(0.3)]
           → Linear(128) → ReLU → [Dropout(0.3)]
           → Linear(26)  [logits]
```

- Вход: изображение 28×28, развёрнутое в вектор длины 784.
- `CrossEntropyLoss` ожидает logits — softmax не применяется в `forward`.
- Dropout включается только опционально (`dropout=True`).

### CNN (Convolutional Neural Network)

```
Input (1×28×28)
  → Conv2d(1→32, 3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)  [32×14×14]
  → Conv2d(32→64, 3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2×2) [64×7×7]
  → Flatten [3136]
  → Linear(3136→256) → ReLU → Dropout(0.4)
  → Linear(256→26)   [logits]
```

- BatchNormalization после каждого свёрточного слоя — стабилизирует обучение.
- MaxPool2d(2×2) вдвое уменьшает пространственное разрешение.
- Dropout(0.4) только перед последним линейным слоем.

---

## Структура проекта

```text
letter-classifier-emnist/
├── notebooks/
│   └── notebooks/Sergey_Smirnov_LetterClassifier.ipynb
├── src/
│   ├── config.py          — гиперпараметры (seed, lr, batch size, epochs)
│   ├── data.py            — загрузка, split, DataLoader для MLP и CNN
│   ├── model_torch.py     — MLP и CNN (PyTorch)
│   ├── model_tf.py        — MLP (TensorFlow/Keras)
│   ├── train_torch.py     — цикл обучения PyTorch + evaluate_torch
│   ├── train_tf.py        — обёртка над model.fit (TensorFlow)
│   ├── evaluation.py      — метрики, графики, confusion matrix, classification report
│   └── inference.py       — predict_letter (PyTorch / TF)
├── custom_images/         — пользовательские изображения для инференса
├── models/                — сохранённые веса (.pth, .h5) [не в git]
├── data/                  — датасет EMNIST [скачивается автоматически, не в git]
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/Serg2311-prog/letter-classifier-emnist.git
cd letter-classifier-emnist
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Запустить ноутбук

```bash
jupyter notebook notebooks/Sergey_Smirnov_LetterClassifier.ipynb
```

Ноутбук запускает все ячейки последовательно:
1. Данные скачиваются автоматически при первом запуске (~500 МБ).
2. Обучение всех моделей займёт ~10–20 минут на CPU (быстрее на GPU).
3. Веса сохраняются в `models/`.

### 4. Инференс на своих изображениях (опционально)

Положите изображение в `custom_images/` и вызовите:

```python
from src.inference import predict_letter

letter, confidence = predict_letter("custom_images/my_letter.png", model, framework="torch")
print(f"Предсказание: {letter}, уверенность: {confidence:.3f}")
```

**Требования к изображению:** любой формат PIL, произвольный размер (ресайз до 28×28 автоматически), буква тёмная на светлом фоне (или наоборот — автоинверсия).

---

## Что реализовано

- Загрузка EMNIST Letters через `torchvision.datasets.EMNIST`
- Конвертация в TensorFlow pipeline через `tf.data.Dataset`
- EDA: распределение классов, визуализация примеров
- Train/validation/test split (90/10 + официальный test)
- **Baseline MLP** на PyTorch и TensorFlow
- **MLP + Dropout** на PyTorch и TensorFlow
- **CNN с BatchNorm** на PyTorch
- Кривые обучения (train + val loss / accuracy)
- Confusion matrix (абсолютная + нормализованная по строкам)
- **Classification report** (precision / recall / F1 по каждому классу)
- **Топ-10 ошибок** с высокой уверенностью
- Сравнительный график всех моделей
- `predict_letter` для пользовательских изображений

---

## Важно

- `data/` — не в git (скачивается автоматически через torchvision).
- `models/` — не в git (файлы весов могут быть большими).
- Все гиперпараметры вынесены в `src/config.py`.
- `SEED = 42` фиксируется во всех генераторах (Python, NumPy, PyTorch, TensorFlow).
