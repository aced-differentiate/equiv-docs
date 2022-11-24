# Guide

Documentation may not be accurate as this is a beta stage package undergoing changes. Raise Github issue if you have questions :)

## Installation
We're a registered Julia package, but it's recommended to install the latest revision directly from Github.
```
using Pkg; Pkg.add(url="https://github.com/aced-differentiate/EquivariantOperators.jl.git")
```
## Scalar & vector fields

Scalar & vector fields are represented as 2d/3d arrays of canonically scalars or vectors (`StaticVectors` from `StaticArrays.jl` for performance). This vector field representation is consistent with multi-channel images from Julia Images which differs from representations using separate arrays for field components. Most `Images` functions are readily applicable. Array values can alternatively be any custom type that Supports addition & multiplication, such as complex numbers and custom structs encoding spherical harmonics.

## Customizable grid, interpolation, particle mesh placement 

```@docs
Grid
place!
```


## Finite difference equivariant operators

```@docs
Del
Lap
Gauss
Op
```

## Convolutions

Operators apply to most use cases but you may also use convolution functions directly. We offer feature rich convolution and cross correlation functions with options for padding, stride, boundary conditions, and custom products (tensor field convolutions). We use `DSP.conv` as our backend for scalar field convolutions and our own implementations for convolutions involving vector fields or custom products. FFT implementation is automatically invoked when appropriate. 

```@docs
cvconv
dspconv
```
