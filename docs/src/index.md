# Home

## Synopsis

EquivariantOperators.jl implements in Julia fully differentiable finite difference operators on scalar or vector fields in 2d/3d. It can run forwards for PDE simulation or image processing, or back propagated for machine learning or inverse problems. Emphasis is on symmetry preserving rotation equivariant operators, including differential operators, common Green's functions & parametrized neural operators. Supports possibly nonuniform, nonorthogonal or periodic grids.

## Tutorials

Hosted on [Colab notebooks](https://colab.research.google.com/drive/17JZEdK6aALxvn0JPBJEHGeK2nO1hPnhQ?usp=sharing)

## Theory

Equivariant linear operators are our building blocks. Equivariance means a rotation of the input results in the same rotation of the output thus preserving symmetry. Applying a linear operator convolves the input with the operator's kernel. If the operator is also equivariant, then its kernel must be radially symmetric. Differential operators and Green's functions are in fact equivariant linear operators. We provide built in constructors for these common operators. By parameterizing the radial function, we can also construct custom neural equivariant operators for machine learning.

## Publications

- [Preprint: Paul Shen, Michael Herbst, Venkat Viswanathan. Rotation Equivariant  Operators for Machine Learning on Scalar Fields and Vector Fields. Arxiv. 2022.](https://arxiv.org/abs/2108.09541)

## Contributors

Lead developer: Paul Shen (xingpins@andrew.cmu.edu), Michael Herbst (herbst@acom.rwth-aachen.de)  
PI: Venkat Viswanathan (venkatv@andrew.cmu.edu)

In collaboration with Julia Computing
