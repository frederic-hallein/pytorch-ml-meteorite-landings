import os.path
import pandas as pd

from .preprocessing import filter_incorrect_data_points

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
        

