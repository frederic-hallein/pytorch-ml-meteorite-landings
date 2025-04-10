import pandas as pd
import numpy as np

from lib.data_analysis.data_transform import degree_to_radians

class TestDataTransform():
    def test_degree_to_radians(self) -> None:
        test_df = pd.DataFrame({
            'col_1': [0.0],
            'col_2': [90.0],
            'col_3': [360.0],
            'col_4': [-90.0],
            'col_5': [540.0],
        })

        col_names = test_df.columns
        actual = degree_to_radians(test_df, col_names)
        expected = pd.DataFrame({
            'col_1': [0.0],
            'col_2': [np.pi / 2],
            'col_3': [2 * np.pi],
            'col_4': [-np.pi / 2],
            'col_5': [3 * np.pi],
        })
        for exp, act in zip(expected.values[0], actual.values[0]):
            assert exp == act, f'Should be equal, but got {exp} == {act} instead.'

    def test_convert_categorical_to_numerical(self) -> None:
        pass