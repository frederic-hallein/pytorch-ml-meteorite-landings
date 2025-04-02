import pandas as pd
import numpy as np

from src.common.helpers import *
from src.neural_networks.neural_network import NeuralNetwork

def main() -> None:
    path = "data/meteorite-landings-filtered.csv"
    df = get_filtered_csv_data(path)

    df = convert_categorical_to_numerical(df)
    print(df.head(10))
    
    input_size = len(df)
    output_size = len(df["recclass"].astype("category").cat.categories)

    # hyperparameters
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 100

    device = get_device()
    model = NeuralNetwork(input_size, output_size).to(device)
    print(model)
    # X = torch.tensor(df["mass"].values, device=device)
    # print(X)


if __name__ == "__main__": main()



    
