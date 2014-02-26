## Quality Control Roadmap

Things to do:

- Extract valid region
- Transform to lock relative coordinates
- Interpolate out zero values (griddata) onto regular grid
- Save this to hdf5

At this point we have the quality controlled data. We then apply the
front relative transformation and *finally* the non dimensionalisation.
