"""
Finite difference calculation & machine learning of electric potential & electric field from charge
"""

using LinearAlgebra
using Plots
using UnPack
include("../src/operators.jl")
include("../src/plotutils.jl")
# using Main.EquivariantOperators # end users should omit Main.

x=reshape(1:25,5,5)
f=ones(3,3)
y=cvconv(x,f;pad=:same,border=:circular)

# make grid
dims = 2
dx = 0.1
cell = dx * Matrix(I, dims, dims)
rmax=.5
grid=Grid(cell,rmax)

# 2d array of f(x,y)=x^3+y^3
@unpack x,y=grid
a=x.^3+y.^3

# make operators
pad=:same
border=:smooth
▽=Del(cell;pad,border)
▽2=Laplacian(cell;pad,border)

del_a=▽(a)
lap_a=▽2(a)

##
plot(
heatmap(a'; title = "x^3+y^3"),
heatmap(lap_a'; title = "x^3+y^3 laplacian"),
vector_field_plot(.1del_a,grid,title = "x^3+y^3 gradient" ),
layout=3)
