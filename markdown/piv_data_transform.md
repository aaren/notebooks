## Front relative transformation

### Basis

The collected velocity data can be transformed to a frame in which
the gravity current is stationary. This is more convenient for data
analysis as the gravity current appears to be a statistically
stationary process.

The coordinates are:

- $x$, the horizontal coordinate, distance from the lock
- $z$, the vertical coordinate, distance from the base
- $t$, the time coordinate, time after lock release
- $t'$, the transformed time coordinate, time after front passage at
  given $x$

The lab coordinates $(x, z, t)$ are transformed into the front
coordinates $(x, z, t')$ by tracking the front of the gravity
current through a run and then subtracting the time of front passage
$t_f$ from the lab time.

$$
t' = t - t_f
$$

```python
from gc_turbulence import SingleLayerRun

test_run = '/home/eeaol/lab/data/flume2/main_data/cache/3mp0zhhn.hdf5'
r = SingleLayerRun(cache_path=test_run)
```

```python
### we are doing this here because we haven't finished the data QC
### yet. 

# get the x and z coords
X = r.x[:, :, 0]
Z = r.z[:, :, 0]
T = r.t[0, 0, :]

# find the indices that correspond to the rectangular view that
# we are going to take into the data
# units in mm
x_min, x_max = -60, 100
z_min, z_max = -100, 20

valid = (X > x_min) & (X < x_max) & (Z > z_min) & (Z < z_max)
iz, ix = np.where(valid)

ix_min, ix_max = ix.min(), ix.max()
iz_min, iz_max = iz.min(), iz.max()

# valid region in x, z
valid = np.s_[iz_min: iz_max, ix_min: ix_max, :]

# example slice in (x, z) (at t=37s)
example_xz = np.s_[iz_min: iz_max, ix_min: ix_max, 700]

# example slice in (t, z) (at x=0)
ix0 = np.argmin(X[0, :] ** 2)
it30 = np.argmin((T - 30) ** 2)
it50 = np.argmin((T - 50) ** 2)
example_tz = np.s_[iz_min: iz_max, ix0, it30: it50]

# example slice in (x, t) (at z=-80)
iz80 = np.argmin((Z[:, 0] + 80) ** 2)
example_xt = np.s_[iz80, ix_min: ix_max, it30: it50]

# example slice (reduced range)
example = np.s_[iz_min: iz_max, ix_min: ix_max, it30: it50]
```

Visually, we take the $(x, t)$ data from each vertical slice, shift
the time values to align with the front passage and then stack all
of the vertical slices.

```python
w_levels = np.linspace(-0.03, 0.04, 100)
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
ax1, ax2, ax3 = axes

ax1.contourf(r.x[example_xz], r.z[example_xz], r.w[example_xz], levels=w_levels)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$z$')
ax1.set_title(r'$(x, z)$ at t=37s')

ax2.contourf(r.t[example_tz], r.z[example_tz], r.w[example_tz], levels=w_levels)
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$z$')
ax2.set_title(r'$(t, z)$ at x=0mm')

ax3.contourf(r.t[example_xt], r.x[example_xt], r.w[example_xt], levels=w_levels)
ax3.set_xlabel(r'$t$')
ax3.set_ylabel(r'$x$')
ax3.set_title(r'$(t, x)$ at z=-80mm')
```


### Front Detection

We want to obtain a function $t_f(x)$ that describes the front
passage time as a function of horizontal position in the camera
frame.

The front position can be seen in the $(x, t)$ plot above as a
clearly defined ridge. We just need to follow the maximum of the
ridge.

```python
contourf(r.t[example_xt], r.x[example_xt], r.w[example_xt], levels=w_levels)
```

The 'proper' way to do this is to convolve something that looks like
the ridge with the ridge over one axis and then pick out the maximum
on that.

The ridge looks roughly like this:

```python
plot(r.t[example_xt][50, :], r.w[example_xt][50, :])
```

Pragmatically, this is such a strong signal that we could just take
the maximum of the raw data.

```python
hovmoller = r.w[example_xt]
maxima_it = np.argmax(hovmoller, axis=1)
maxima_ix = np.indices(maxima_it.shape).squeeze()

maxima_space = r.x[example_xt][maxima_ix, maxima_it]
maxima_time = r.t[example_xt][maxima_ix, maxima_it]

contourf(r.t[example_xt], r.x[example_xt], r.w[example_xt], levels=w_levels)
plot(maxima_time, maxima_space, 'go')
xlabel('time after lock release (s)')
ylabel('horizontal position (mm)')
```

There are some outliers. We could reject them or we can strengthen
the initial detection. Let's take a column average of the vertical
velocity and look for departure from zero.

```python
mean_data = r.w[example].mean(axis=0)
lines = plot(r.t[example][0, :, :].T, mean_data.T)
```

```python
column_avg = r.w[valid].mean(axis=0)
exceed = column_avg > 0.01
# find first occurence of exceeding 0.01
front_it = np.argmax(exceed, axis=1)
front_ix = np.indices(front_it.shape).squeeze()

front_space = r.x[example_xt][front_ix, front_it]
front_time = r.t[example_xt][front_ix, front_it]

contourf(r.t[example_xt], r.x[example_xt], r.w[example_xt], levels=w_levels)
plot(front_time, front_space, 'go')
xlabel('time after lock release (s)')
ylabel('horizontal position (mm)')
```

That's pretty ideal.

Let's apply it to everything and reshape the data:

```python
t_ = r.t[valid] - front_time.reshape((1, -1, 1))

T_ = t_[0, 0, :]
it_min = np.argmin((T_ + 5) ** 2)
it_max = np.argmin((T_ - 20) ** 2)
# careful! we've already taken the valid slice
# so this stacks on top of that.
example_xt_ = np.s_[iz80, :, it_min: it_max]

contourf(t_[example_xt_], r.x[valid][example_xt_], r.w[valid][example_xt_], levels=w_levels)
axvline(0)
xlabel('time after front passage (s)')
ylabel('horizontal position (mm)')
```

Whilst this seems to give us what we want, we need to to a proper
transform to get a regular grid with consistent limits over each of
the axes.


### Transforming the data

We need to interpolate this to a rectangular grid to make it
sensible. Since our original data is on a regular grid we can use
`scipy.ndimage.map_coordinates` to do this. All we have to do is
workout what the time coordinates are of the front relative system.
The $(x, z)$ coordinates remain the same.

`map_coordinates` takes an input array and an array of coordinates
as arguments, and returns the input array evaluated at the
coordinates (in array space).

*You can essentially think of map_coordinates as a way to index an
array with non integer indices.*

The shape of the output is determined by dropping the first axis of
the coordinates, i.e. coordinates must have shape (ndim, output_shape).

Here's a simple example where we get output == input:

```python
import scipy.ndimage as ndi

a = np.arange(12).reshape((4, 3))
out = ndi.map_coordinates(a, np.indices(a.shape))
assert((out == a).all())
```

In our case we want to sample the original data over the original x
and z, but with a time sampling from $t_f + t_0$ to $t_f + t_1$.

We have $t_f$ as a 1d array, both real (`front_time`) and grid
(`front_it`, the index in t at which front passes, over all x).

```python
# get the real start time of the data and the
# sampling distance in time (dt)
rt = r.t[valid][0, 0, :]
dt = rt[1] - rt[0]

## compute the times at which to sample the *front relative*
## data, if it already existed as a 3d rectangular array (which it
## doesn't!). The other way to do it would be to compute the
## necessary sample indices first and then index the r.t array.
## You shouldn't do it that way because the coords can be negative
## floats and map_coordinates will handle that properly.
#
# start and end times (s) relative to front passage
t0 = -5
t1 = 20
relative_sample_times = np.arange(t0, t1, dt)
# extend over x and z
sz, sx, = r.t[valid].shape[:2]
relative_sample_times = np.tile(relative_sample_times, (sz, sx, 1))

## now compute the times at which we need to sample the 
## original data to get front relative data by adding
## the time of front passage* onto the relative sampling times
## *(as a function of x)
rtf = front_time[None, ..., None] + relative_sample_times

# grid coordinates of the sampling times
# (has to be relative to what the time is at
# the start of the data).
t_coords = (rtf - rt[0]) / dt

# z and x coords are the same as before
zx_coords = np.indices(t_coords.shape)[:2]

# required shape of the coordinates array is (3, rz.size, rx.size, rt.size)
coords = np.concatenate((zx_coords, t_coords[None]), axis=0)

st = t_coords.shape[-1]
X_ = r.x[valid][:, :, 0, None].repeat(st, axis=-1)
Z_ = r.z[valid][:, :, 0, None].repeat(st, axis=-1)
T_ = relative_sample_times
W_ = ndi.map_coordinates(r.w[valid], coords)

# N.B. there is an assumption here that r.t, r.z and r.x are
# 3d arrays. They are redundant in that they repeat over 2 of
# their axes (r.z, r.x, r.t = np.meshgrid(z, x, t, indexing='ij'))
```

Now we can plot, e.g. the front relative mean:

```python
# front relative example
example_fr_tz = s_[:, 50, :]
example_fr_tx = np.s_[iz80, :, :]

# mean over horizontal
W_bar = np.mean(W_, axis=1)

fig, axes = plt.subplots(nrows=3, figsize=(8, 6))
axes[0].set_title(r'$w(t, x)$')
axes[0].set_ylabel('horizontal (mm)')

axes[1].set_title(r'$w(t, z)$')
axes[1].set_ylabel('vertical (mm)')

axes[2].set_title(r'$\barw_t(t, z)$')
axes[2].set_ylabel('vertical (mm)')
axes[2].set_xlabel('time after front passage (s)')

c0 = axes[0].contourf(T_[example_fr_tx], X_[example_fr_tx], W_[example_fr_tx], levels=w_levels)
c1 = axes[1].contourf(T_[example_fr_tz], Z_[example_fr_tz], W_[example_fr_tz], levels=w_levels)
c2 = axes[2].contourf(T_[example_fr_tz], Z_[example_fr_tz], W_bar, levels=w_levels)
```
