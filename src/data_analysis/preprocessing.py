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
    
    # Process each column
    for col_name in col_names:
        # Get unique categories and store them
        original_categories = df[col_name].unique()
        
        # Create mapping dictionary for this column
        category_mappings[col_name] = dict(zip(range(len(original_categories)), original_categories))
        
        # Convert to category and then to numerical
        df[col_name] = df[col_name].astype("category")
        df[col_name] = df[col_name].cat.codes
    
    return df, category_mappings