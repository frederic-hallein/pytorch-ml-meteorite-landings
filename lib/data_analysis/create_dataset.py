import pandas as pd
import torch
from torch.utils.data import TensorDataset

def create_dataset(df: pd.DataFrame, feature_names: list[str], label_name: str) -> TensorDataset:
    """
    Create dataset from pandas dataframe.

    :param df: pd.DataFrame contaning data
    :param feature_names: List of feature names
    :param label_name: String name label
    :return: TensorDataset of dataframe
    """
    df = df.dropna() # drop all rows that contain NaN values
    features = torch.tensor(df[feature_names].values, dtype=torch.float)
    labels   = torch.tensor(df[[label_name]].values, dtype=torch.float)
    dataset  = TensorDataset(features, labels)
    return dataset