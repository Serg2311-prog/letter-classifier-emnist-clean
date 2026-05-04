import torch
from torch import nn

from .config import INPUT_SIZE, NUM_CLASSES, IMAGE_SIZE


class MLP(nn.Module):
    """MLP для EMNIST Letters.

    Важно: softmax не используется в forward, потому что CrossEntropyLoss
    ожидает logits.
    """

    def __init__(self, dropout: bool = False, dropout_rate: float = 0.3):
        super().__init__()

        def make_reg():
            return nn.Dropout(dropout_rate) if dropout else nn.Identity()

        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, 512),
            nn.ReLU(),
            make_reg(),

            nn.Linear(512, 256),
            nn.ReLU(),
            make_reg(),

            nn.Linear(256, 128),
            nn.ReLU(),
            make_reg(),

            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def count_torch_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CNN(nn.Module):
    """Свёрточная нейронная сеть для EMNIST Letters.

    Вход: тензор формы (batch, 1, 28, 28).
    Два conv-блока (Conv→BN→ReLU→MaxPool), затем классификатор MLP.
    Softmax не применяется — CrossEntropyLoss ожидает logits.
    """

    def __init__(self, dropout_rate: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 28×28 → 14×14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 14×14 → 7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 64 × 7 × 7 = 3136 features after two 2×2 max-pools on 28×28 input
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
