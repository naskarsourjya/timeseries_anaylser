import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing library
from timeseries import regular_ts, irregular_ts


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

# frequency response
np.random.seed(42)
fs = 100
t  = np.arange(0, 2, 1 / fs)

df = pd.DataFrame(
    {f"test{i+1}": (np.sin(2 * np.pi * 2 * t) +
                    0.5 * np.sin(2 * np.pi * 10 * t) +
                    0.1 * np.random.randn(len(t)))
    for i in range(10)},
    index=t)

ts_dummy3 = regular_ts('Dummy Time series 3')
ts_dummy3.data = df
ts_dummy3.dt=1 / fs

_, _= ts_dummy3.plot_distribution()
#plt.show()

#plt.show()
fig3, ax3=ts_dummy3.plot_bode_distribution()
ts_dummy3_filtered = ts_dummy3.lowpass_filter(cutoff_hz=10)
_, _= ts_dummy3_filtered.plot_bode_distribution(color='green', fig=fig3, ax=ax3, title="Tuning Filter")
plt.show()
pass


