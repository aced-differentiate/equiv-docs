# Operators basics, machine learning, inverse problems

```@example 1
DIR = joinpath(pwd(), "../../src") # change this to your EquivariantOperators.jl directory
"""
Finite difference calculation & machine learning of electric potential & electric field from charge
"""

using LinearAlgebra
using Plots
using Random
using Flux
Random.seed!(1)
include("$DIR/operators.jl")

# make grid
dims = 2
dx = 0.1
cell = dx * I(dims)
rmax = 1.0
grid = Grid(cell, rmax)

# make operators
rmin = 1e-9
rmax = sqrt(3)
ϕ = Op(r -> 1 / (4π * r), rmin, rmax, cell)
E = Op(r -> 1 / (4π * r^2), rmin, rmax, cell; l = 1)
▽ = Op(:▽, cell)

# put dipole charges
ρf = zeros(size(grid)..., 1)
put!(ρf, grid, [0.5 0.0; -0.5 0.0]', [1.0 -1]'')

# calculate fields
Ef = E(ρf)
ϕf = ϕ(ρf)

# test
rvec = [0, 0]
@show get(ϕf, grid, rvec), [0.0]
@show get(Ef, grid, rvec), get(-▽(ϕ(ρf)), grid, rvec), [-2 / (4π * 0.5^2), 0]

p = []
push!(p, heatmap(ρf[:, :, 1]',title = "dipole charge"), )
push!(p, heatmap(ϕf[:, :, 1]', title = "dipole potential"),)
plot(p..., layout = length(p))
savefig("d1.svg");

# @unpack x, y=grid
x = grid.coords[:, :, 1]
y = grid.coords[:, :, 2]
s = 1e-2
u = s * Ef[:, :, 1]
v = s * Ef[:, :, 2]
x, y, u, v = vec.([x, y, u, v])
quiver(x, y, quiver = (u, v), c = :blue, title = "dipole electric field")
savefig("d2.svg");

##
# make neural operators
ϕ_ = Op(Radfunc(), rmin, rmax, cell)
E_ = Op(Radfunc(), rmin, rmax, cell; l = 1)

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
p = []
push!(p, heatmap(ϕ_.kernel[:, :, 1], title = "learned potential kernel"))
r = 0:0.01:1
push!(
    p,
    plot(r, ϕ_.radfunc.(r), title = "learned potential kernel radial function"),
)
plot(p..., layout = length(p))
savefig("ml1.svg");

x = E_.grid.coords[:, :, 1]
y = E_.grid.coords[:, :, 2]
s = 1e-2
u = s * E_.kernel[:, :, 1]
v = s * E_.kernel[:, :, 2]
x, y, u, v = vec.([x, y, u, v])
quiver(x, y, quiver = (u, v), c = :blue, title = "learned E field kernel ")
savefig("ml2.svg");

```

![](d1.svg)
![](d2.svg)
![](ml1.svg)
![](ml2.svg)
