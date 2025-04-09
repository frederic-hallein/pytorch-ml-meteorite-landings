import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.nn_layer_sequence = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Prevent overfitting
            nn.Linear(10, num_classes),
            # nn.Softmax(dim=1)   # Convert to probabilities
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn_layer_sequence(x)