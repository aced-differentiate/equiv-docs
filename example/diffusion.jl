"""
simulation & machine learning of diffusion advection PDE
"""

using IterTools
using Plots
using Random
using LinearAlgebra
Random.seed!(1)
include("../src/operators.jl")
# using EquivariantOperators


# make grid
n = 2
dx = 0.02
cell = dx * Matrix(I, n, n)
rmax = 1.0
grid = Grid(cell, rmax)
sz = size(grid)
▽ = Del(cell)
△ = Lap(cell)
# laplacian
# blur=Op(:Gauss,cell;σ=.1)

# diffusion advection with point emitter
p = [0.2, 0.5, 0.5, 0.8]
D, vx, vy, k = p
v = fill([vx, vy], sz)
u0 = zeros(sz)
s = zeros(sz)
place!(s, grid, [0.0, 0.0], 1.0)
# s=blur(s)
f(u, p, t) = D * (△(u)) - v .· ▽(u) .+ s * k

# simulate PDE
using DifferentialEquations
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)
sol = DifferentialEquations.solve(prob, Tsit5(), reltol = 1e-3, abstol = 1e-3)

# plot
using Plots
gr()
anim = Animation()

t = 0:0.02:1
for t in t
    heatmap(sol(t)[:, :, 1], clim = (0, 10))
    frame(anim)
end
gif(anim, "f.gif", fps = 10)

##
data = [(sol(t), f(sol(t), 0, 0)) for t in LinRange(0, 1, 10)]
# op = Op(Radfunc(),-1e-6, 2dx, cell)
# ps=Flux.params(op)
p_ = ones(length(p))
ps = Flux.params(p_)

function loss(u, du)
    D, vx, vy, k = p_
    v = fill([vx, vy], sz)
    duhat = D * (△(u)) - v .· ▽(u) .+ s * k
    @show l = nae(duhat, du)
end

loss(data[1]...)
opt = ADAM(0.1)

Flux.@epochs 2 Flux.train!(loss, ps, data, opt)
@show p, p_
# (p, p_) = ([0.2, 0.5, 0.5, 0.8], [0.2113613015968995, 0.4977070976688276, 0.4977070976688435, 0.791498663145966])

# # loss(data[1]...)
#
# using GalacticOptim,Optim
# using OptimizationOptimJL
# # using OptimizationOptimisers
#
# function loss(p,_)
#     u, du=data[1]
#     D,vx,vy,k=p
#     v = fill([vx,vy],sz)
#     duhat= D * (△(u)) - v .⋅ ▽(u).+s*k
#     @show l = nae(duhat, du)
# end
#
# opt = Optim.BFGS()
# # opt = Adam()
# l = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
# prob = OptimizationProblem(l, p_, data)
# sol = GalacticOptim.solve(prob, opt, maxiters = 10)
#
# @show p,p_
# # (p, p_) = ([0.2, 0.5, 0.5, 0.8], [0.2113613015968995, 0.4977070976688276, 0.4977070976688435, 0.791498663145966])
