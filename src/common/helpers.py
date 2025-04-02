import os.path
import torch
import pandas as pd
import matplotlib.pyplot as plt

def _filter_csv_data() -> pd.DataFrame:
    """
    Filter incorect data points:
    (1) dates before 860 CE or after 2016 are incorrect -> should actually be BCE years
    (2) entries have latitude and longitude of 0N/0E -> western coast of Africa, 
    where it would be quite difficult to recover meteorites
    """
    path = "data/archive/meteorite-landings.csv"
    if not os.path.isfile(path):
        raise AssertionError(f"File '{path}' does not exist.")

    df = pd.read_csv(path)

    for x in df.index:
        if df.loc[x, "year"] <= 860 or df.loc[x, "year"] >= 2016 \
            or (df.loc[x, "reclong"] == 0.0 and df.loc[x, "reclat"] == 0.0) \
            or (df.loc[x, "reclong"] < -180.0 or df.loc[x, "reclong"] > 180.0):
            df.drop(x, inplace = True)

    new_path = "data/meteorite-landings-filtered.csv"
    df.to_csv(new_path, index=False)
    return df

def get_filtered_csv_data(path: str) -> pd.DataFrame:
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        print(f"File '{path}' does not exist. Create new filtered dataset.")
        df = _filter_csv_data()
    return df

def convert_categorical_to_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the categorical data in 'nametype', 'fall' and 'reclass' to numerical
    """
    df["nametype"] = df["nametype"].astype("category")
    df["fall"] = df["fall"].astype("category")
    # df["recclass"] = df["recclass"].astype("category")
    # print(df.select_dtypes(['category']).columns)
    # print (len(df['nametype'].cat.categories))
    # print (len(df['fall'].cat.categories))
    # print (len(df['recclass'].cat.categories))

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

def get_device() -> str:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return device

def _plot_pairplot(df: pd.DataFrame) -> None:
    pd.plotting.scatter_matrix(df, figsize=(10, 10),  hist_kwds = {'bins': 20}, s = 10, alpha = 0.8)
    plt.savefig("plots/pairplot_numerical_data.png")


def _plot_geolocation(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Geolocation of meteorites", fontsize = 16)
    ax.scatter(
        x=df["reclong"] , 
        y=df["reclat"],
        marker='.',
        color="blue"
    )
    ax.set(
        xlabel = "reclong [rad]",
        ylabel = "reclat [rad]"
    )
    
    plt.savefig("plots/meteorite_geolocation.png")

def plot_data(df: pd.DataFrame) -> None:
    num_df = df[["mass", "year", "reclat", "reclong"]]
    _plot_pairplot(num_df)
    _plot_geolocation(df)
    plt.show()
