import torch

# TODO : use evaluation dataset
def evaluate(model, test_dataset, category_map, device) -> None:
    """
    Evaluate the model using an evaluation dataset.

    :param model: ...
    :param evaluation_dataset: ...
    :param category_map: ...
    :param device: String name of device
    """
    print('\nModel Evaluation:')
    model.eval()
    x, y = test_dataset[0][0], test_dataset[0][1]
    with torch.no_grad():
        x = x.to(torch.float)
        x = x.to(device)
        pred = model(x)
        predicted, actual = category_map[torch.argmax(pred).item()], category_map[y.item()]
        print(f"Predicted: '{predicted}', \n   Actual: '{actual}'")