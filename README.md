# Spatial Information in Dimensionality Reduction (Spidr)

To clone the repo and some dependencies (hnsw lib):

```git clone --recurse-submodule https://github.com/alxvth/SpidrPlugin.git```

Build with the same generator as the HDPS core.

## Other Dependencies
Not all dependencies are included in this repo, some need to be downloaded/install by yourself. 
Make sure to adjust your system variables respectively:
- HDPS core, follow instructions [here](https://github.com/hdps/core)
- Qt 5.14
- Boost (headers-only, define the system variable BOOST_INCLUDEDIR for cmake to find it automatically)
- OpenMP
