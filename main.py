import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class irregular_ts():

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


class regular_ts():

    __array_ufunc__ = None

    def __init__(self, name='Regular Time Series'):
        self.name = name
        self.data = pd.DataFrame()
        self.shift = 0
        self.dt = None

    def set_time(self, time):
        assert type(time) is np.ndarray, f"time must be a numpy array. Type found: {type(time)}"
        assert time.ndim <= 2, f"time must be a 1-D or a flat 2-D array. Shape found: {time.shape}"
        if time.ndim == 2:
            assert time.shape[1] == 1, f"time must be a 1-D or a column 2-D array. Shape found: {time.shape}"
            time = time.flatten()
        
        # check if all the differences between consecutive time points are the same
        time_diff = np.diff(time.flatten())
        assert np.all(time_diff == time_diff[0]), "Time points must be evenly spaced"

        # storing time as the index of the dataframe
        self.data.index = time
        self.dt = time_diff[0]

        # eof
        return None


    def append_data(self, new_data, series_names):

        assert self.data.index.shape[0] != 0, "Set time with set_time() before calling append_data()!"
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

        # storing new data
        new_data_df = pd.DataFrame(new_data, columns=series_names, index=self.data.index)

        # joining new data with old data
        self.data = self.data.join(new_data_df, how="outer")

        # eof
        return None


    def z_shift(self, shift):

        assert isinstance(shift, int), f"shift must me an integer. Type found: {type(shift)}"
        
        # creating new object
        new_regular_ts = regular_ts(name=self.name)
        new_regular_ts.data = self.data.copy()
        new_regular_ts.shift = shift
        new_regular_ts.dt = self.dt

        return new_regular_ts


    def _shift_lossless(self, r_ts):

        # init
        df = r_ts.data.copy()
        neg_shift = -r_ts.shift
        dt = r_ts.dt


        if neg_shift>0:
            df_nan_index = np.arange(df.index[-1] + dt, df.index[-1]+(neg_shift+1)*dt, dt)
            df_nan = pd.DataFrame(data = np.full((neg_shift, len(df.columns)), np.nan), columns=df.columns,
                                  index=df_nan_index)
            df_nonshifted = pd.concat([df, df_nan])
            df_new = df_nonshifted.shift(neg_shift)
            pass

        elif neg_shift==0:
            df_new = df

        else:
            df_nan_index = np.arange(df.index[0] + dt*(neg_shift), df.index[0], dt)
            df_nan = pd.DataFrame(data=np.full((-neg_shift, len(df.columns)), np.nan), columns=df.columns,
                                  index=df_nan_index)
            df_nonshifted = pd.concat([df_nan, df])
            df_new = df_nonshifted.shift(neg_shift)
            pass

        return df_new.copy()

    def plot_tests(self, fig=None, ax=None, title=None) -> plt.Figure:

        # init
        df = self._shift_lossless(r_ts=self)
        t = df.index.to_numpy()
        mean = df.mean(axis=1)
        q05 = df.quantile(0.05, axis=1)
        q95 = df.quantile(0.95, axis=1)
        ylabel = self.name

        # generating new title if needed
        if title is None:
            title = f"{self.name} Distribution vs Time"

        # initiating new plot is needed
        if fig==None or ax==None:
            fig, ax = plt.subplots(figsize=(10, 5))

        # ── Individual tests ─────────────────────────────────────────────────
        for i, col in enumerate(df.columns):
            ax.plot(t, df[col], color="steelblue", alpha=0.3, linewidth=0.8,
                    label="Individual tests" if i == 0 else "")

        # ── Quantile band ────────────────────────────────────────────────────
        ax.fill_between(t, q05, q95, color="steelblue", alpha=0.15,
                        label="5th–95th percentile")

        # ── Mean ─────────────────────────────────────────────────────────────
        ax.plot(t, mean, color="navy", linewidth=2, label="Mean")
        ax.plot(t, q05, color="darkorange", linewidth=1.2, linestyle="--", label="5th quantile")
        ax.plot(t, q95, color="darkorange", linewidth=1.2, linestyle="--", label="95th quantile")

        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax


    # algebraic operations
    def __add__(self, other):
        
        # if both are regular_ts, then we can add them directly
        if isinstance(other, regular_ts):
            assert self.dt == other.dt, (f"Time steps (dt) of the two regular_ts objects must be the same. Found: "
                                         f"{self.dt} and {other.dt}")
            assert self.data.columns.equals(other.data.columns), (f"Series names of the two regular_ts objects must "
                                                                  f"be the same. Found: {self.data.columns} and "
                                                                  f"{other.data.columns}")
            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)
            df2_shifted = self._shift_lossless(r_ts=v2)

            # addition
            df_add = df1_shifted + df2_shifted
            df_new = df_add.dropna()

            #  new regular ts
            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = 0
            new_regular_ts.dt = self.dt
        
        # if other is a scalar
        elif isinstance(other, (int, float)):

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # addition
            df_add = df1_shifted + v2
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        # if other is a numpy array
        elif isinstance(other, np.ndarray):

            assert other.shape == (1,1), f"other must be a 2-D array with only 1 element. Shape found: {other.shape}"

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # addition
            df_add = df1_shifted + np.full((df1_shifted.shape[0], df1_shifted.shape[1]), v2.flatten()[0])
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        else:
            raise ValueError(f"Unsupported type for addition: {type(other)}. Supported types are: regular_ts, int, "
                             f"float, np.ndarray")

        # eof
        return new_regular_ts

    def __radd__(self, other):

        return self.__add__(other=other)
    

    def __sub__(self, other):

        # if both are regular_ts, then we can add them directly
        if isinstance(other, regular_ts):
            assert self.dt == other.dt, (f"Time steps (dt) of the two regular_ts objects must be the same. Found: "
                                         f"{self.dt} and {other.dt}")
            assert self.data.columns.equals(other.data.columns), (f"Series names of the two regular_ts objects must "
                                                                  f"be the same. Found: {self.data.columns} and "
                                                                  f"{other.data.columns}")
            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)
            df2_shifted = self._shift_lossless(r_ts=v2)

            # subtraction
            df_add = df1_shifted - df2_shifted
            df_new = df_add.dropna()

            #  new regular ts
            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = 0
            new_regular_ts.dt = self.dt

        # if other is a scalar
        elif isinstance(other, (int, float)):

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # subtraction
            df_add = df1_shifted - v2
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        # if other is a numpy array
        elif isinstance(other, np.ndarray):

            assert other.shape == (1, 1), f"other must be a 2-D array with only 1 element. Shape found: {other.shape}"

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # subtraction
            df_add = df1_shifted - np.full((df1_shifted.shape[0], df1_shifted.shape[1]), v2.flatten()[0])
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        else:
            raise ValueError(f"Unsupported type for subtraction: {type(other)}. Supported types are: regular_ts, int, "
                             f"float, np.ndarray")

        # eof
        return new_regular_ts
    

    def __rsub__(self, other):

        if isinstance(other, (int, float)):
            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # subtraction
            df_add = v2 - df1_shifted
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt
        
        elif isinstance(other, np.ndarray):
            assert other.shape == (1, 1), f"other must be a 2-D array with only 1 element. Shape found: {other.shape}"

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # subtraction
            df_add = np.full((df1_shifted.shape[0], df1_shifted.shape[1]), v2.flatten()[0]) - df1_shifted
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        else:
            raise ValueError(f"Unsupported type for subtraction: {type(other)}. Supported types are: regular_ts, int, "
                             f"float, np.ndarray")
        

    def __mul__(self, other):

        # if both are regular_ts, then we can add them directly
        if isinstance(other, regular_ts):
            assert self.dt == other.dt, (f"Time steps (dt) of the two regular_ts objects must be the same. Found: "
                                         f"{self.dt} and {other.dt}")
            assert self.data.columns.equals(other.data.columns), (f"Series names of the two regular_ts objects must "
                                                                  f"be the same. Found: {self.data.columns} and "
                                                                  f"{other.data.columns}")
            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)
            df2_shifted = self._shift_lossless(r_ts=v2)

            # multiplication
            df_add = df1_shifted * df2_shifted
            df_new = df_add.dropna()

            #  new regular ts
            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = 0
            new_regular_ts.dt = self.dt

        # if other is a scalar
        elif isinstance(other, (int, float)):

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # multiplication
            df_add = df1_shifted * v2
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        # if other is a numpy array
        elif isinstance(other, np.ndarray):

            assert other.shape == (1, 1), f"other must be a 2-D array with only 1 element. Shape found: {other.shape}"

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # multiplication
            df_add = df1_shifted * np.full((df1_shifted.shape[0], df1_shifted.shape[1]), v2.flatten()[0])
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        else:
            raise ValueError(f"Unsupported type for multiplication: {type(other)}. Supported types are: regular_ts, int, "
                             f"float, np.ndarray")

        # eof
        return new_regular_ts


    def __rmul__(self, other):

        return self.__mul__(other=other)

    
    def __truediv__(self, other):
        
        # if both are regular_ts, then we can add them directly
        if isinstance(other, regular_ts):
            assert self.dt == other.dt, (f"Time steps (dt) of the two regular_ts objects must be the same. Found: "
                                         f"{self.dt} and {other.dt}")
            assert self.data.columns.equals(other.data.columns), (f"Series names of the two regular_ts objects must "
                                                                  f"be the same. Found: {self.data.columns} and "
                                                                  f"{other.data.columns}")
            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)
            df2_shifted = self._shift_lossless(r_ts=v2)

            # division
            df_add = df1_shifted / df2_shifted
            df_new = df_add.dropna()

            #  new regular ts
            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = 0
            new_regular_ts.dt = self.dt
        
        # if other is a scalar
        elif isinstance(other, (int, float)):

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # division
            df_add = df1_shifted / v2
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        # if other is a numpy array
        elif isinstance(other, np.ndarray):

            assert other.shape == (1,1), f"other must be a 2-D array with only 1 element. Shape found: {other.shape}"

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # division
            df_add = df1_shifted / np.full((df1_shifted.shape[0], df1_shifted.shape[1]), v2.flatten()[0])
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        else:
            raise ValueError(f"Unsupported type for division: {type(other)}. Supported types are: regular_ts, int, "
                             f"float, np.ndarray")

        # eof
        return new_regular_ts
    

    def __rtruediv__(self, other):

        if isinstance(other, (int, float)):
            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # division
            df_add = v2 / df1_shifted
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

            pass
        
        elif isinstance(other, np.ndarray):
            assert other.shape == (1, 1), f"other must be a 2-D array with only 1 element. Shape found: {other.shape}"

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # division
            df_add = np.full((df1_shifted.shape[0], df1_shifted.shape[1]), v2.flatten()[0]) / df1_shifted
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        else:
            raise ValueError(f"Unsupported type for division: {type(other)}. Supported types are: regular_ts, int, "
                             f"float, np.ndarray")

        # eof
        return new_regular_ts

    
    def __pow__(self, other):
        
        # if both are regular_ts, then we can add them directly
        if isinstance(other, regular_ts):
            assert self.dt == other.dt, (f"Time steps (dt) of the two regular_ts objects must be the same. Found: "
                                         f"{self.dt} and {other.dt}")
            assert self.data.columns.equals(other.data.columns), (f"Series names of the two regular_ts objects must "
                                                                  f"be the same. Found: {self.data.columns} and "
                                                                  f"{other.data.columns}")
            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)
            df2_shifted = self._shift_lossless(r_ts=v2)

            # power
            df_add = df1_shifted ** df2_shifted
            df_new = df_add.dropna()

            #  new regular ts
            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = 0
            new_regular_ts.dt = self.dt

        # if other is a scalar
        elif isinstance(other, (int, float)):

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # power
            df_add = df1_shifted ** v2
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        # if other is a numpy array
        elif isinstance(other, np.ndarray):

            assert other.shape == (1, 1), f"other must be a 2-D array with only 1 element. Shape found: {other.shape}"

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # power
            df_add = df1_shifted ** np.full((df1_shifted.shape[0], df1_shifted.shape[1]), v2.flatten()[0])
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        else:
            raise ValueError(f"Unsupported type for power: {type(other)}. Supported types are: regular_ts, int, "
                             f"float, np.ndarray")

        # eof
        return new_regular_ts
    

    def __rpow__(self, other):

        if isinstance(other, (int, float)):
            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # power
            df_add = v2 ** df1_shifted
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        elif isinstance(other, np.ndarray):
            assert other.shape == (1, 1), f"other must be a 2-D array with only 1 element. Shape found: {other.shape}"

            # init
            v1 = self
            v2 = other

            # shifting operations
            df1_shifted = self._shift_lossless(r_ts=v1)

            # power
            df_add = np.full((df1_shifted.shape[0], df1_shifted.shape[1]), v2.flatten()[0]) ** df1_shifted
            df_new = df_add.dropna()

            new_regular_ts = regular_ts(name=self.name)
            new_regular_ts.data = df_new
            new_regular_ts.shift = self.shift
            new_regular_ts.dt = self.dt

        else:
            raise ValueError(f"Unsupported type for power: {type(other)}. Supported types are: regular_ts, int, "
                             f"float, np.ndarray")

        # eof
        return new_regular_ts


    # representation
    def __repr__(self):
        return f"regular_ts: {self.name}\nData Shape:\n{self.data.shape}\nShift: {self.shift}\nTime Step (dt): {self.dt}"

# debug statements
t_dummy = np.hstack(5*[5*np.arange(0, 10).reshape(-1, 1)])
y_dummy = np.hstack(5*[10*np.arange(0, 10).reshape(-1, 1)])
ts_dummy = irregular_ts(name="Dummy Series")
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

ts_dummy.append_data(new_time=np.array([99, 101]), new_data=np.array([199, 200]), series_names='final2')
ts_dummy.data.head()


ts_dummy2 = regular_ts(name='Dummy Series 2')
#ts_dummy2.set_time(time=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1))
ts_dummy2.set_time(time=5*np.arange(0,10))
ts_dummy2.append_data(new_data=y_dummy,
                      series_names=[f'series{n+1}' for n in range(5)])
ts_dummy2.append_data(new_data=0.5*y_dummy,
                      series_names=[f'series{n+6}' for n in range(5)])

print(ts_dummy2.data.head())

#new_ts = ts_dummy2.z_shift(shift=1) + ts_dummy2.z_shift(shift=-2)


# mathematical operations
# additions
t1 = ts_dummy2.z_shift(shift=-1) + ts_dummy2.z_shift(shift=+2)
t2 = ts_dummy2.z_shift(shift=-1) + 3.5
t3 = ts_dummy2.z_shift(shift=-1) + np.array([[3.5]])

# reverse additions
#_ = ts_dummy2.z_shift(shift=-1) + np.full((10, 10), 6)
t4 =  3.5 + ts_dummy2.z_shift(shift=-1)
t5 = np.array([[3.5]]) + ts_dummy2.z_shift(shift=-1)

# subtractions
t6 = ts_dummy2.z_shift(shift=-1) - ts_dummy2.z_shift(shift=+2)
t7 = ts_dummy2.z_shift(shift=-1) - 3.5
t8 = ts_dummy2.z_shift(shift=-1) - np.array([[6]])

# reverse subtractions
#_ = ts_dummy2.z_shift(shift=-1) - ts_dummy2.z_shift(shift=+2)
t9 = - 3.5 - ts_dummy2.z_shift(shift=-1)
t10 = - np.array([[6]]) - ts_dummy2.z_shift(shift=-1)

# multiply
t11 = ts_dummy2.z_shift(shift=-1) * ts_dummy2.z_shift(shift=+2)
t12 = ts_dummy2.z_shift(shift=-1) * 3.5
t13 = ts_dummy2.z_shift(shift=-1) * np.array([[3.5]])

# reverse multiply
t14 = 3.5 * ts_dummy2.z_shift(shift=-1)
t15 = np.array([[3.5]]) * ts_dummy2.z_shift(shift=-1)

# division
t16 = ts_dummy2.z_shift(shift=-1) / ts_dummy2.z_shift(shift=+2)
t17 = ts_dummy2.z_shift(shift=-1) / 2
t18 = ts_dummy2.z_shift(shift=-1) / np.array([[2]])

# reverse division
t19 = 3.5 / ts_dummy2.z_shift(shift=-1)
t20 = np.array([[3.5]]) / ts_dummy2.z_shift(shift=-1)
print(t20)

# plots
fig, _= ts_dummy2.plot_tests()
plt.show()
pass


