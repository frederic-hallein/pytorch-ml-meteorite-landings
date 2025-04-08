from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        # self.nn_layer_sequence = nn.Sequential(
        #     nn.Linear(input_size, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.Softmax(dim=1),
        #     nn.Linear(32, num_classes)
        # )
        self.nn_layer_sequence = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Prevent overfitting
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)   # Convert to probabilities
        )

    def forward(self, x) -> None:
        return self.nn_layer_sequence(x)