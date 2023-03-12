# Spatial Information in Dimensionality Reduction (Spidr) Plugin

To clone the repo and some dependencies (hnsw lib):

```git clone --recurse-submodule https://github.com/hdps/SpidrPlugin.git```

Build with the same generator as the HDPS core, see instructions [here](https://github.com/hdps/core).

Uses the Spidr implementation from [here](https://github.com/alxvth/Spidr/), which build on top of the A-tSNE implemention from the [HDILib](https://github.com/biovault/HDILib).

## Building

The HDILib dependency is currently handled a little different depending on which OS you want to build the Spidr library.

On Linux, the HDILib is build with this project. vcpkg will handle all it's dependencies.

On Windows, you can do the same, but when inlcuding this project into another, the resulting library might have some dll dependencies on lz4 (a flann dependency, which in turn in used by HDILib). This is easier to handle with the [HDILibSlim](https://github.com/alxvth/HDILibSlim), a version of the HDILib without dependencies. Build it, set `BUILD_HDILIB` OFF and `USE_HDILIBSLIM` ON, and provide the `HDILIBSLIM_ROOT` to the cmake folder in the HDILibSlim\lib installation directory. 
