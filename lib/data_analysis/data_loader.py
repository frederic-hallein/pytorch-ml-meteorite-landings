import os.path
import pandas as pd

def filter_incorrect_data_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset that contains the following
    incorrectly parsed entries:
    - Date that is before 860 CE or after 2016
    - Latitude and longitude of 0N/0E

    :param df: pd.DataFrame containing the dataset
    :return: pd.DataFrame containing filtered dataset
    """
    for x in df.index:
        if df.loc[x, 'year'] <= 860 or df.loc[x, 'year'] >= 2016 \
            or (df.loc[x, 'reclong'] == 0.0 and df.loc[x, 'reclat'] == 0.0) \
            or (df.loc[x, 'reclong'] < -180.0 or df.loc[x, 'reclong'] > 180.0):
            df.drop(x, inplace = True)

    return df

def get_filtered_csv_data(path_filtered_data: str) -> pd.DataFrame:
    """
    Get filtered csv data.

    :param path_filtered_data: String name of path to filtered data
    :return: pd.DataFrame containing filtered dataset
    """
    if not os.path.isfile(path_filtered_data):
        print(f"File '{path_filtered_data}' does not exist. Creating new filtered dataset from original dataset.")
        path_original_data = "data/archive/meteorite-landings.csv"
        try:
            df = pd.read_csv(path_original_data)
        except FileNotFoundError:
            raise AssertionError(f"File '{path_original_data}' does not exist.")

        df = filter_incorrect_data_points(df)
        df.to_csv(path_filtered_data, index=False)

    return pd.read_csv(path_filtered_data)


