import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.data_analysis import get_filtered_csv_data, convert_categorical_to_numerical, plot_data
from src.machine_learning import train, test, evaluate
from src.machine_learning import NeuralNetwork

# Paths
PATH       = 'data/archive/meteorite-landings-filtered.csv'
MODEL_PATH = 'models/model.pth'

# Variables
CAT_COL       = ['nametype', 'recclass', 'fall']
FEATURE_NAMES = ['nametype', 'mass', 'fall', 'year', 'reclat', 'reclong']
LABEL_NAME    = 'recclass'

# Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE    = 64
EPOCHS        = 10

def main() -> None:
    df = get_filtered_csv_data(PATH)
    # plot_data(df) # TODO : make more diverse plots to enhance understanding of data

    df, category_mappings = convert_categorical_to_numerical(df, CAT_COL)

    input_size  = len(FEATURE_NAMES)
    num_classes = len(df[LABEL_NAME].astype('category').cat.categories)

    # drop all rows that contain NaN values
    df = df.dropna()

    features = torch.tensor(df[FEATURE_NAMES].values, dtype=torch.float)
    labels   = torch.tensor(df[[LABEL_NAME]].values, dtype=torch.float)
    dataset  = TensorDataset(features, labels)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    device = torch.accelerator.current_accelerator().type \
             if torch.accelerator.is_available() else 'cpu'
    print(f'Using {device} device')

    model = NeuralNetwork(input_size, num_classes).to(device)
    print(f'Using the following model:\n{model}\n')

    train_dataloader = DataLoader(train_dataset, BATCH_SIZE)
    test_dataloader  = DataLoader(test_dataset , BATCH_SIZE)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    for t in range(EPOCHS):
        print(f"Epoch { t + 1 }\n-------------------------------")
        train(train_dataloader, device, model, loss_function, optimizer)
        test(test_dataloader, device, model, loss_function)
    print('Model training and testing finished.')

    # save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved model state to '{MODEL_PATH}'")

    # model evaluation
    recclass_map = category_mappings[LABEL_NAME]
    evaluate(model, test_dataset, recclass_map, device)

if __name__ == "__main__": main()