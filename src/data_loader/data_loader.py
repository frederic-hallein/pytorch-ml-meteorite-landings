import os.path
import pandas as pd

def create_filtered_csv_data(path: str) -> None:
    """
    Create new filterd csv dataset.
    The following incorrectly parsed entries are filtered:
    - Date that is before 860 CE or after 2016
    - Latitude and longitude of 0N/0E

    :param path: String name of path
    :return: None
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise AssertionError(f"File '{path}' does not exist.")

    for x in df.index:
        if df.loc[x, "year"] <= 860 or df.loc[x, "year"] >= 2016 \
            or (df.loc[x, "reclong"] == 0.0 and df.loc[x, "reclat"] == 0.0) \
            or (df.loc[x, "reclong"] < -180.0 or df.loc[x, "reclong"] > 180.0):
            df.drop(x, inplace = True)

    path_filtered_data = "data/meteorite-landings-filtered.csv"
    df.to_csv(path_filtered_data, index=False)

def get_filtered_csv_data(path: str) -> pd.DataFrame:
    """ 
    Get filtered csv data.

    :param path: String name of path
    :return: pd.DataFrame containing filtered dataset
    """
    if not os.path.isfile(path):
        print(f"File '{path}' does not exist. Creating new filtered dataset from original dataset.")
        path_original_data = "data/archive/meteorite-landings.csv" 
        create_filtered_csv_data(path_original_data)
        
    return pd.read_csv(path)

