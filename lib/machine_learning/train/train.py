import torch
from torch.utils.data import DataLoader

def train(dataloader: DataLoader, device: str, model, loss_function, optimizer) -> None:
    """
    Train the model using a training dataset.

    :param dataloader: Dataloader of training dataset
    :param device: String name of device
    :param model: model used for training
    :param loss_function: loss function used for backward propagation
    :param optimizer: optimizer used for optimizing model parameters
    """
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(torch.float)
        y = torch.flatten(y.to(torch.long))
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_function(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")