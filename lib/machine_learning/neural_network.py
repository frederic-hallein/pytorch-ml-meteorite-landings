import torch
from torch import nn
class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.nn_layer_sequence = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn_layer_sequence(x)