import torch
from torch import optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src.helper.helper import convert_categorical_to_numerical
from src.data_loader.data_loader import get_filtered_csv_data
from src.neural_network.neural_network import NeuralNetwork

def train(dataloader, model, loss_fn, optimizer, device) -> None:
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(torch.float)
        y = torch.flatten(y.to(torch.long))
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device) -> None:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(torch.float)
            y = torch.flatten(y.to(torch.long))
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct   += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main() -> None:
    path = "data/meteorite-landings-filtered.csv"
    df = get_filtered_csv_data(path)
    # plot_data(df)

    input_size = 6
    num_classes = len(df["recclass"].astype("category").cat.categories)

    cat_col = ['nametype', 'recclass', 'fall']
    df = convert_categorical_to_numerical(df, cat_col)

    df = df.dropna() #drop all rows that have any NaN values

    features = torch.tensor(df[['nametype', 'mass', 'fall', 'year', 'reclat', 'reclong']].values)
    labels   = torch.tensor(df[['recclass']].values)
    dataset  = TensorDataset(features, labels)

    device = torch.accelerator.current_accelerator().type \
             if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork(input_size, num_classes).to(device)
    print(model)

    # hyperparameters
    learning_rate = 0.001
    batch_size    = 64
    epochs        = 5

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size)
    test_dataloader  = DataLoader(test_dataset , batch_size)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    #save model
    model_path = "models/model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to '{model_path}'")

    # TODO : first check if model already exists
    # load model
    model = NeuralNetwork(input_size, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Loaded PyTorch Model State from '{model_path}'")


if __name__ == "__main__": main()