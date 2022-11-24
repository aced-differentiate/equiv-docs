# Home

## Synopsis

EquivariantOperators.jl implements in Julia fully differentiable finite difference and Green's function operators on scalar or vector fields in any dimension. Utilities for grid interpolation, particle mesh methods, boundary conditions and convolutions are also included. It can run forwards for FDTD simulation or image processing, or back propagated for machine learning or inverse problems. Emphasis is on symmetry preserving rotation equivariant operators, including differential operators, common Green's functions & parametrized neural operators. Supports scalar and vector field convolutions with customizable products eg  `dot`. Automatically performs convolutions using FFT when it's faster doing so. Supports possibly nonuniform, nonorthogonal or periodic grids.

## Features

- Array operators in any dimension including derivative, gradient, divergence, Laplacian, curl, and customizable Green's functions
- Interpolation and particle mesh methods on possibly non-orthogonal grids
- Boundary conditions including periodic, smooth, zero, mirror
- Feature rich convolutions on scalar arrays and vector fields with automatic FFT computation when appropriate
- Fully differentiable compatible with `Flux` `Zygote`

## Github
Star us at [Github](https://github.com/aced-differentiate/EquivariantOperators.jl) :) Raise Github issue if you have questions :)

## Theory

Equivariant linear operators are our building blocks. Equivariance means a rotation of the input results in the same rotation of the output thus preserving symmetry. Applying a linear operator convolves the input with the operator's kernel. If the operator is also equivariant, then its kernel must be radially symmetric. Differential operators and Green's functions are in fact equivariant linear operators. We provide built in constructors for these common operators. By parameterizing the radial function, we can also construct custom neural equivariant operators for machine learning.

## Publications

Preprint outdated, releasing new version soon
- [Preprint: Paul Shen, Michael Herbst, Venkat Viswanathan. Rotation Equivariant  Operators for Machine Learning on Scalar Fields and Vector Fields. Arxiv. 2022.](https://arxiv.org/abs/2108.09541)

## Contributors

Lead developer: Paul Shen (pxshen@alumni.stanford.edu)

Collaborators: Michael Herbst (herbst@acom.rwth-aachen.de)  

PI: Venkat Viswanathan (venkatv@andrew.cmu.edu)

In collaboration with Julia Computing
