import numpy as np
import pandas as pd

def standardization(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Standardize the dataset.

    :param df: pd.DataFrame containing the dataset
    :param features: List of feature names
    :return: pd.DataFrame containing standardized dataset
    """
    df[features] -= df[features].mean()
    df[features] /= df[features].std()
    return df

def degree_to_radians(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Covert degrees to radians.

    :param df: pd.DataFrame containing the dataset
    :param features: List of feature names
    :return: pd.DataFrame containing values in radians
    """
    df[features] = df[features] * np.pi / 180
    return df

def convert_categorical_to_numerical(df: pd.DataFrame, col_names: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Convert the categorical data into numerical and return the mapping dictionary.

    :param df: pd.DataFrame containing the dataset
    :param col_names: List of column names
    :return: Tuple containing:
                - pd.Dataframe containing converted dataset
                - dict containing categorical mappings
    """
    category_mappings = {}
    for col_name in col_names:
        original_categories = df[col_name].unique()
        category_mappings[col_name] = dict(zip(range(len(original_categories)), original_categories))

        df[col_name] = df[col_name].astype("category")
        df[col_name] = df[col_name].cat.codes

    return df, category_mappings
