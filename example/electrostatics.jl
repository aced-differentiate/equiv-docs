"""
Finite difference calculation & machine learning of electric potential & electric field from charge
"""

using LinearAlgebra
using Plots
using Random
using Flux
Random.seed!(1)
include("../src/operators.jl")
include("../src/plotutils.jl")
# using Main.EquivariantOperators # end users should omit Main.

# make grid
dims = 2
dx = 0.1
cell = dx * Matrix(I, dims, dims)
rmax = 1.0
grid = Grid(cell, rmax)

# make operators
rmin = 1e-9
pad = :same
ϕ = Op(r -> 1 / (4π * r), rmax, cell; rmin, pad)
E = Op(r -> 1 / (4π * r^2), rmax, cell; rmin, pad, l = 1)
▽ = Del(cell)

# put dipole charges
ρf = zeros(size(grid))
put!(ρf, grid, [0.5, 0.0], 1)
put!(ρf, grid, [-0.5, 0.0], -1)

# calculate fields
Ef = E(ρf)
ϕf = ϕ(ρf)

# test
rvec = [0, 0]
@show get(ϕf, grid, rvec), [0.0]
@show get(Ef, grid, rvec), get(-▽(ϕf), grid, rvec), [-2 / (4π * 0.5^2), 0]


##
# make neural operators
nbasis = 32
ϕ_ = Op(Radfunc(nbasis), rmax, cell; rmin, pad)
E_ = Op(Radfunc(nbasis), rmax, cell; rmin, l = 1, pad)

ps = Flux.params(ϕ_, E_)
function loss()
    remake!(E_)
    remake!(ϕ_)
    global E_f = E_(ρf)
    global ϕ_f = ϕ_(ρf)
    @show l = (nae(E_f, Ef) + nae(ϕ_f, ϕf)) / 2
    l
end

data = [()]
loss()
opt = ADAM(0.1)
Flux.@epochs 50 Flux.train!(loss, ps, data, opt)

## plot
r = 0:0.02:1

# layout = @layout [
#     # a{0.4h}
#     Plots.grid(1,3)
#     Plots.grid(2,2)
# ]
plot(
    heatmap(ρf', title = "dipole charge"),
    heatmap(ϕf', title = "dipole potential"),
    vector_field_plot(0.1Ef, grid; title = "dipole electric field");
    layout = 3,
)

plot(
    heatmap(ϕ_.kernel, title = "learned potential kernel"),
    plot(r, ϕ_.radfunc.(r), title = "learned potential kernel radial function"),
    vector_field_plot(
        1 * E_.kernel,
        grid;
        title = "learned electric field kernel ",
    ),
    plot(r, E_.radfunc.(r), title = "learned electric field radial function");
    layout = (2,2), )
##
