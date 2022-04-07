# Spatial Information in Dimensionality Reduction (Spidr)

To clone the repo and some dependencies (hnsw lib):

```git clone --recurse-submodule https://github.com/hdps/SpidrPlugin.git```

Build with the same generator as the HDPS core., see instructions [here](https://github.com/hdps/core).

Uses the Spidr implementation from [here](https://github.com/alxvth/Spidr/), which build on top of the A-tSNE implemention from the [HDILib](https://github.com/biovault/HDILib).

## Other Dependencies
By default all dependencies for Spidr are included in this repo or automatically downloaded. See the [build instructions](https://github.com/alxvth/Spidr/) of the Spidr library for information on it's dependencies (HDILib, Boost and Eigen). 
