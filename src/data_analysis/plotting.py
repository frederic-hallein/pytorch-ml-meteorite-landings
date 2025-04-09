import pandas as pd
import matplotlib.pyplot as plt

def _plot_pairplot(df: pd.DataFrame) -> None:
    """
    Plot and save a pairplot of the data.

    :param df: pd.DataFrame containing the dataset
    :return: None
    """
    pd.plotting.scatter_matrix(df, figsize=(10, 10),  hist_kwds = {'bins': 20}, s = 10, alpha = 0.8)
    plt.savefig("plots/pairplot_numerical_data.png")


def _plot_geolocation(df: pd.DataFrame) -> None:
    """
    Plot and save the geolocation of meteorite data.
    - reclong == longitude
    - reclat  == lattitude

    :params df: pd.DataFrame containing the dataset
    :return: None
    """
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
    """
    Plot and show the data.

    :params df: pd.DataFrame containing the dataset
    :return: None
    """
    # num_df = df[["mass", "year", "reclat", "reclong"]]
    # _plot_pairplot(df)
    _plot_geolocation(df)
    plt.show()