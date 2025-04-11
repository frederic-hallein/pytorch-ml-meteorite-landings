import torch
from torch import nn
from torch.utils.data import DataLoader

from lib.machine_learning.train import train
from lib.machine_learning.test import test

def create_ml_model(model: any,
                    train_dataset: torch.utils.data.dataset.Subset,
                    test_dataset: torch.utils.data.dataset.Subset,
                    learning_rate: float,
                    batch_size: int,
                    epochs: int,
                    device: str
    ) -> any:
    """
    Create machine learning model.

    :param model: NeuralNetwork model
    :param train_dataset: Torch subset of dataset for training
    :param test_dataset: Torch subset of dataset for testing
    :param learnin_rate: Float value of learning rate
    :param batch_size: Integer value of batch size
    :param epochs: Integer value of epochs
    :param device: String name of device
    :return: NeuralNetwork model
    """
    # model = NeuralNetwork(input_size, num_classes).to(device)
    print(f'Using the following model:\n{model}\n')
    print(f"Learning rate = {learning_rate}")
    print(f"Batch size    = {batch_size}")
    print(f"Epochs        = {epochs}")

    train_dataloader = DataLoader(train_dataset, batch_size)
    test_dataloader  = DataLoader(test_dataset , batch_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    for t in range(epochs):
        print(f"\nEpoch { t + 1 }\n-------------------------------")
        train(train_dataloader, device, model, loss_function, optimizer)
        test(test_dataloader, device, model, loss_function)
    print('Model training and testing finished.')
    return model