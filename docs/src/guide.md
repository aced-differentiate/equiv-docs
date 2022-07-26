# Guide

Documentation may not be accurate as this is a beta stage package undergoing changes. Get started with [tutorials](https://colab.research.google.com/drive/17JZEdK6aALxvn0JPBJEHGeK2nO1hPnhQ) which are more up to date.

## Scalar & vector fields

Scalar & vector fields are represented as 2d/3d arrays of canonically scalars or vectors. Array values can alternatively be any type that Supports addition & multiplication.

## Customizable grid

```@docs
Grid
```

## Particle mesh placement and interpolation

```@docs
get(::AbstractArray, ::Grid, ::AbstractVector)
```

## Finite difference operators

```@docs
Op
```

## Lower level utilities

### Convolutions

Feature rich convolution and cross correlation functions with options for padding, stride, boundary conditions, and custom products (tensor field convolutions).

```@docs
cvconv
dspconv
```
