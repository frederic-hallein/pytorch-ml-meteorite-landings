import pandas as pd
import matplotlib.pyplot as plt

def _plot_geolocation(df: pd.DataFrame) -> None:
    """
    Plot and save the geolocation of meteorite data.
    - reclong == longitude
    - reclat  == lattitude

    :params df: pd.DataFrame containing the dataset
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle('Geolocation of Meteorites', fontsize = 16)
    ax.scatter(
        x=df['reclong'] ,
        y=df['reclat'],
        marker='.',
        color="blue"
    )
    ax.set(
        xlabel = 'reclong [degree]',
        ylabel = 'reclat [degree]'
    )

    plt.savefig('plots/meteorite_geolocation.png')

def _plot_recclass(df: pd.DataFrame) -> None:
    """
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle('Classes of Meteorites', fontsize = 16)
    ax = df['recclass'].value_counts().plot(kind='barh')
    ax.invert_yaxis()
    ax.set(
        ylim=(15,-1),
        xlabel='recclass',
        ylabel='count',
    )

    plt.savefig("plots/meteorite_recclass.png")

def plot_data(df: pd.DataFrame) -> None:
    """
    Plot and show the data.

    :params df: pd.DataFrame containing the dataset
    :return: None
    """

    _plot_geolocation(df)
    _plot_recclass(df)
    plt.show()