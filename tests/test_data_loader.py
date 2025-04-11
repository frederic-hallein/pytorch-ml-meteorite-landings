import pandas as pd

from lib.data_analysis.data_loader import filter_incorrect_data_points

class TestDataLoader():
    def test_filter_incorrect_data_points(self) -> None:
        year = []
        reclong = []
        reclat = []

        # Case 1: Pre-860 CE dates
        year.extend([800, 859, 861])
        reclong.extend([10.0, 0.0, 20.0])
        reclat.extend([10.0, 0.0, 20.0])

        # Case 2: Post-2016 dates
        year.extend([2015, 2016, 2017])
        reclong.extend([15.0, 0.0, 25.0])
        reclat.extend([15.0, 0.0, 25.0])

        # Case 3: 0N/0E coordinates
        year.extend([1000, 1500, 2000])
        reclong.extend([0.0, 0.0, 0.0])
        reclat.extend([0.0, 0.0, 0.0])

        # Case 4: Invalid longitude ranges
        year.extend([1200, 1400, 1600])
        reclong.extend([-190.0, 190.0, 181.0])
        reclat.extend([30.0, 40.0, 50.0])

        # Create DataFrame
        test_df = pd.DataFrame({
            'year'   : year,
            'reclong': reclong,
            'reclat' : reclat,
        })

        actual = filter_incorrect_data_points(test_df)
        expected = pd.DataFrame({
            'year'   : [861, 2015],
            'reclong': [ 20,   15],
            'reclat' : [ 20,   15]
        })

        for exp, act in zip(expected.values[0], actual.values[0]):
            assert exp == act, f'Should be equal, but got {exp} == {act} instead.'
