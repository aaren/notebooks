Probability distribution of gravity current piv data
----------------------------------------------------

We would like to know whether gravity currents exhibit any
predictable structure in the distribution of their winds. Potential
applications of this knowledge are in dust uplift and engineering.


### Basic Setup

```python
import scipy.stats as stats

import gc_turbulence as g

pf = '/nfs/see-fs-02_users/eeaol/lab/data/flume2/main_data/cache/3mp0xba3_processed.hdf5'
r = g.ProcessedRun(pf)
r.load()
```

To get an idea of the basic structure, here is the average and
standard deviation of the vertical velocity in the gravity current
relative system:

```python
figure(figsize=(16, 8))
W_mean = np.mean(r.W_, axis=1)
W_std = np.std(r.W_, axis=1)
nan_std = stats.nanstd(r.W_, axis=1)

w_levels = np.linspace(-0.02, 0.035, 100)

fig, ax = plt.subplots(nrows=3)
ax[0].set_title('Mean')
c0 = ax[0].contourf(W_mean, levels=w_levels)
ax[1].set_title('Std deviation')
c1 = ax[1].contourf(W_std, levels=np.linspace(0, 0.02, 100))
ax[2].set_title('nan Std deviation')
c2 = ax[2].contourf(nan_std, levels=np.linspace(0, 0.02, 100))

fig.colorbar(c0, ax=ax[0], use_gridspec=True)
fig.colorbar(c1, ax=ax[1], use_gridspec=True)
fig.colorbar(c2, ax=ax[2], use_gridspec=True)
```

I'm not sure what the artifacts are in the standard deviation plots.
Plotting by excluding nans doesn't fix the issue. This suggests to
me that we need to filter the data - perhaps a convolution filter?


### Statistics

Pre-filter the data:

```python
def no_zero(array):
    return array[array != 0]
```

Data from a single vertical slice:

```python
data = r.W_[:, 50, 100]
z = r.Z_[:, 50, 100]
plot(data, z)
xlabel('vertical velocity')
ylabel('height')
```

From multiple slices, but the same time after front passage:

```python
data = r.W_[:, :, 100]
z = r.Z_[:, :, 100]
plot(data, z, 'k', linewidth=0.3)
xlabel('vertical velocity')
ylabel('height')
```

As a pdf:

```python
fig, ax = plt.subplots()
w_range = w_levels.min() / 10, w_levels.max() / 10

nbins = 200
bin_values = np.linspace(w_range[0], w_range[1], nbins)
hist_z_100 = np.array([np.histogram(no_zero(z), range=w_range,
                                    bins=nbins, density=True)[0]
                        for z in r.W_[:, :, 100:110]])
heights = r.Z_[:, 0, 0]
ax.contourf(bin_values, heights, hist_z_100, levels=np.linspace(0, 800))
```

Data from a single *height*, through time:

```python
data = r.W_[30, 50, :]
t = r.T_[30, 50, :]
plot(t, data)
xlabel('time after front passage')
ylabel('vertical velocity')
```

From multiple slices, but the same height:

```python
figure(figsize=(16, 8))
data = r.W_[30, :, :]
t = r.T_[30, :, :]
plot(t.T, data.T, 'k', linewidth=0.1)
xlabel('time after front passage')
ylabel('vertical velocity')
ylim(-0.04, 0.04)
```

As a pdf:

```python
fig, ax = plt.subplots()
w_range = w_levels.min() / 10, w_levels.max() / 10

nbins = 200
bin_values = np.linspace(w_range[0], w_range[1], nbins)
hist_z_100 = np.array([np.histogram(no_zero(t), range=w_range,
                                    bins=nbins, density=True)[0]
                        for t in r.W_[30, :, :].T])
times = r.T_[0, 0, :]
ax.contourf(bin_values, times, hist_z_100, levels=np.linspace(0, 800))
```

### Dynamic mode decomposition

```python
dmd = g.analysis.DMD()
modes = dmd.calculate_dmd(r.W_[...], n_modes=20)

dmd.plot_dmd(modes, r.X_, r.Z_, r.T_, r.W_)
```
