import numpy as np
import pandas as pd

class timeseries():

    def __init__(self, name):
        self.name = name
        self.data = pd.DataFrame()
        self.series_names = []

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

        for i, name in enumerate(series_names):
            new_data_df = pd.DataFrame(new_data[:, i], columns=[name], index=new_time[:, i].flatten())
            self.data = self.data.join(new_data_df, how="outer")
            #self.data = pd.concat([self.data, new_data_df])


t_dummy = np.hstack(5*[5*np.arange(0, 10).reshape(-1, 1)])
y_dummy = np.hstack(5*[10*np.arange(0, 10).reshape(-1, 1)])
ts_dummy = timeseries(name="Dummy Series")
names=list(range(y_dummy.shape[1]))
ts_dummy.append_data(new_time=t_dummy, new_data=y_dummy, series_names=names)
ts_dummy.data.head()

t_dummy2 = np.hstack(2*[6*np.arange(0, 10).reshape(-1, 1)])
y_dummy2 = np.hstack(2*[12*np.arange(0, 10).reshape(-1, 1)])
names=[6, 7]
ts_dummy.append_data(new_time=t_dummy2, new_data=y_dummy2, series_names=names)
ts_dummy.data.head()

ts_dummy.append_data(new_time=np.array([0, 100]), new_data=np.array([50, 150]), series_names='final')
ts_dummy.data.head()