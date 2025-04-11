import torch
from torch.utils.data import DataLoader

def test(dataloader: DataLoader, device: str, model, loss_function) -> None:
    """
    Test the model using a test dataset.

    :param dataloader: Dataloader of test dataset
    :param device: String name of device
    :param model: model used for testing
    :param loss_function: loss function used to calculate average loss
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y = torch.flatten(y.to(torch.long))
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct   += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct   /= size
    print(f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")