"""
Finite difference calculation & machine learning of electric potential & electric field from charge
"""

using LinearAlgebra
using Plots
using UnPack
include("../src/operators.jl")
include("../src/plotutils.jl")
# using Main.EquivariantOperators # end users should omit Main.

x = reshape(1:25, 5, 5)
f = ones(3, 3)
y = cvconv(x, f; pad=:same, border=:circular)

# make grid
dx = dy= 0.1
resolutions = (dx, dy)
rmax = 0.5
grid = Grid(resolutions, rmax)

@unpack x, y = grid
a = x .^ 3 + y .^ 3 # 2d array

# make operators
▽ = Del(resolutions)
▽2 = Lap(resolutions)

del_a = ▽(a)
lap_a = ▽2(a)

##
xaxis=yaxis=-0.5:dx:0.5
plot(
    heatmap(xaxis,yaxis,a'; title="x^3+y^3"),
    heatmap(xaxis,yaxis,lap_a'; title="x^3+y^3 laplacian"),
    # vector_field_plot(.1del_a,grid,title = "x^3+y^3 gradient" ),
    layout=2)
