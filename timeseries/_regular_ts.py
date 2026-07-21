import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from scipy import signal


class regular_ts:

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


        if neg_shift >0:
            df_nan_index = np.arange(df.index[-1] + dt, df.index[-1 ] +(neg_shift +1 ) *dt, dt)
            df_nan = pd.DataFrame(data = np.full((neg_shift, len(df.columns)), np.nan), columns=df.columns,
                                  index=df_nan_index)
            df_nonshifted = pd.concat([df, df_nan])
            df_new = df_nonshifted.shift(neg_shift)
            pass

        elif neg_shift==0:
            df_new = df

        else:
            df_nan_index = np.arange(df.index[0] + dt *neg_shift, df.index[0], dt)
            df_nan = pd.DataFrame(data=np.full((-neg_shift, len(df.columns)), np.nan), columns=df.columns,
                                  index=df_nan_index)
            df_nonshifted = pd.concat([df_nan, df])
            df_new = df_nonshifted.shift(neg_shift)
            pass

        return df_new.dropna().copy()


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

            assert other.shape == (1 ,1), f"other must be a 2-D array with only 1 element. Shape found: {other.shape}"

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
            raise ValueError \
                (f"Unsupported type for multiplication: {type(other)}. Supported types are: regular_ts, int, "
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

            assert other.shape == (1 ,1), f"other must be a 2-D array with only 1 element. Shape found: {other.shape}"

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
        return (f"regular_ts: {self.name}\nData Shape:\n{self.data.shape}\nShift: {self.shift}\nTime Step (dt): "
                f"{self.dt}")

    def plot_distribution(self, fig=None, ax=None, title=None, plot_all=True,
                          figsize=(10, 5), color="steelblue"):
        """
        Plot time-domain distribution with consistent color shading.

        Parameters
        ----------
        color : str or hex
            Base color. Shades are derived automatically:
            - Mean             → darkest
            - Quantile lines   → moderate
            - Individual ts    → light
            - Shaded area      → lightest
        """
        # ── Derive shades from base color ────────────────────────────────────
        c_mean = self._adjust_lightness(color, 0.5)  # darkest
        c_quantile = self._adjust_lightness(color, 0.7)  # moderate
        c_individual = self._adjust_lightness(color, 1.3)  # light
        c_area = self._adjust_lightness(color, 1.6)  # lightest

        # ── Init ─────────────────────────────────────────────────────────────
        df = self._shift_lossless(r_ts=self)
        t = df.index.to_numpy()
        mean = df.mean(axis=1)
        q05 = df.quantile(0.05, axis=1)
        q95 = df.quantile(0.95, axis=1)
        ylabel = self.name

        if title is None:
            title = f"{self.name} Distribution vs Time"

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # ── Individual trajectories (light) ──────────────────────────────────
        if plot_all:
            for i, col in enumerate(df.columns):
                ax.plot(t, df[col], color=c_individual, alpha=0.6, linewidth=0.8,
                        label="Individual ts" if i == 0 else "")

        # ── Shaded quantile area (lightest) ──────────────────────────────────
        ax.fill_between(t, q05, q95, color=c_area, alpha=1.0,
                        label="5th–95th percentile")

        # ── Quantile lines (moderate) ─────────────────────────────────────────
        ax.plot(t, q05, color=c_quantile, linewidth=1.2, linestyle="--", label="5th quantile")
        ax.plot(t, q95, color=c_quantile, linewidth=1.2, linestyle="--", label="95th quantile")

        # ── Mean (darkest) ────────────────────────────────────────────────────
        ax.plot(t, mean, color=c_mean, linewidth=2, label="Mean")

        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def _adjust_lightness(self, color, amount):
        """
        amount < 1 → darker
        amount > 1 → lighter
        """
        try:
            c = mcolors.cnames[color]
        except KeyError:
            c = color
        c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


    def plot_bode_distribution(self, fig=None, ax=None, title=None,
                               figsize=(10, 5), color="steelblue"):
        """
        Plot Bode diagram with consistent color shading.

        Parameters
        ----------
        color : str or hex
            Base color. Shades are derived automatically:
            - Mean          → darkest
            - Quantile lines → moderate
            - Individual ts  → light
            - Shaded area    → lightest
        """

        # ── Derive shades from base color ────────────────────────────────────
        c_mean = self._adjust_lightness(color, 0.5)  # darkest
        c_quantile = self._adjust_lightness(color, 0.7)  # moderate
        c_individual = self._adjust_lightness(color, 1.3)  # light
        c_area = self._adjust_lightness(color, 1.6)  # lightest

        # ── Init ─────────────────────────────────────────────────────────────
        df = self._shift_lossless(r_ts=self)
        dt = self.dt
        N = df.shape[0]
        if title is None:
            title = f"Bode Plot of: {self.name}"

        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize)

        ax_mag = ax[0]
        ax_phase = ax[1]

        mags = {}
        phases = {}

        # ── Compute FFT per column ───────────────────────────────────────────
        for col in df.columns:
            y = df[col].to_numpy()
            Y = np.fft.rfft(y - y.mean())
            freqs = np.fft.rfftfreq(N, d=dt)
            Y, freqs = Y[1:], freqs[1:]

            mags[col] = 20 * np.log10(np.abs(Y))
            phases[col] = np.unwrap(np.angle(Y)) * 180 / np.pi

        df_mag = pd.DataFrame(mags, index=freqs)
        df_phase = pd.DataFrame(phases, index=freqs)

        # ── Plot ─────────────────────────────────────────────────────────────
        for df_plot, ax, ylabel in [
            (df_mag, ax_mag, "Magnitude (dB)"),
            (df_phase, ax_phase, "Phase (degrees)")
        ]:
            mean = df_plot.mean(axis=1)
            q05 = df_plot.quantile(0.05, axis=1)
            q95 = df_plot.quantile(0.95, axis=1)

            # Individual trajectories
            for i, col in enumerate(df_plot.columns):
                ax.semilogx(freqs, df_plot[col], color=c_individual, alpha=0.6,
                            linewidth=0.8, label="Individual ts" if i == 0 else "")

            # Shaded quantile area (lightest)
            ax.fill_between(freqs, q05, q95, color=c_area, alpha=1.0,
                            label="5th–95th percentile")

            # Quantile lines (moderate)
            ax.semilogx(freqs, q05, color=c_quantile, linewidth=1.2,
                        linestyle="--", label="5th quantile")
            ax.semilogx(freqs, q95, color=c_quantile, linewidth=1.2,
                        linestyle="--", label="95th quantile")

            # Mean (darkest)
            ax.semilogx(freqs, mean, color=c_mean, linewidth=2, label="Mean")

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)

        ax_mag.set_title(title)
        return fig, [ax_mag, ax_phase]


    def lowpass_filter(self, cutoff_hz: float, order: int = 4):

        # init
        df = self._shift_lossless(r_ts=self)
        dt = self.dt
        fs = 1.0 / dt
        nyquist = fs / 2.0
        df_filtered = pd.DataFrame(columns=df.columns, index=df.index)

        assert cutoff_hz < nyquist, (f"cutoff_hz ({cutoff_hz} Hz) must be below "
                                     f"Nyquist frequency ({nyquist:.2f} Hz)")

        normalized_cutoff = cutoff_hz / nyquist

        # design Butterworth filter
        b, a = signal.butter(order, normalized_cutoff, btype="low")

        # apply filter to each column
        for col in df_filtered.columns:
            df_filtered[col] = signal.filtfilt(b, a, df[col].to_numpy())

        # return a new regular_ts with filtered data
        new_regular_ts = regular_ts(name=self.name)
        new_regular_ts.data = df_filtered
        new_regular_ts.dt = self.dt
        new_regular_ts.shift = 0

        # eof
        return new_regular_ts
