from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.nn_layer_sequence = nn.Sequential(
            nn.Linear(input_size, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.Softmax(dim=1),
            nn.Linear(6, output_size)
        )

    def forward(self, x) -> None:
        return self.nn_layer_sequence(x)