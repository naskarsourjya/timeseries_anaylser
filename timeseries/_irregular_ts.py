import numpy as np
import pandas as pd


class irregular_ts:

    def __init__(self, name='Irregular Time Series'):
        self.name = name
        self.data = pd.DataFrame()


    def append_data(self, new_time, new_data, series_names):
        assert type(new_time) is np.ndarray, "new_time must be a numpy array"
        assert new_time.shape == new_data.shape, "new_time and new_data must have the same shape"
        if new_time.ndim == 1:
            new_time = new_time.reshape((-1, 1))
        elif new_time.ndim > 2:
            raise ValueError("new_time must be a 1D or 2D array")

        assert type(new_data) is np.ndarray, "new_data must be a numpy array"
        if new_data.ndim == 1:
            new_data = new_data.reshape((-1, 1))
            if type(series_names) == str:
                series_names = [series_names]
        elif new_data.ndim == 2:
            assert len(series_names) == new_data.shape[
                1], "Length of series_names must match number of columns in new_data"
        elif new_data.ndim > 2:
            raise ValueError("new_data must be a 1D or 2D array")

        # is the series name is already in self.data, then we will throw an error
        for name in series_names:
            if name in self.data.columns:
                raise ValueError(f"Series name '{name}' already exists in the data. Please use a unique name.")

        for i, name in enumerate(series_names):
            new_data_df = pd.DataFrame(new_data[:, i], columns=[name], index=new_time[:, i].flatten())
            self.data = self.data.join(new_data_df, how="outer")