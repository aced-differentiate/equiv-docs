"""
We demonstrate core capabilities of EquivariantOperators.jl: particle mesh, finite differences, and machine learning. In a particle mesh context, we place 2 +1 point charges in a grid creating a scalar field. We then do a finite difference calculation of the electric field and potential. Switching gears to machine learning, we train our equivariant neural network to learn this transformation from the charge distribution (scalar field) to the electric potential (scalar field) and field (vector field), essentially learning the Green's function solution to Poisson's Equation. Finally, we treat our point charges as proton nuclei and train another neural network to predict the ground state electronic density of H2.
"""

include("../src/operators.jl")
using Flux
using Zygote
# using Enzyme
using BlackBoxOptim
using Images

using FileIO
using Plots
using Random
Random.seed!(1)

name = "h2"
case = load("..\\density-prediction\\data\\$name.jld2", "cases")[1]
@show positions = case.positions
@show charges = case.charges
natoms = length(charges)

# grid params
@show dx = case.resolution
# @show origin = case.origin
@show origin = ones(3)
ρe = -abs.(case.density)

s = 4
# ρp=ρp[1:s:end,1:s:end,1:s:end]
# ρe=ρe[1:s:end,1:s:end,1:s:end]
ρe = imresize(ρe, ratio=1 / s)
dx *= s

sz = size(ρe)
cell = dx * Matrix(I, 3, 3)
grid = Grid(cell, sz; origin)
@unpack dv = grid

C = sum(ρe) * dv
@show C, -sum(charges)

ρp = zeros(sz)
place!(ρp, grid, positions, charges)

function normρe(ρe, dv, C)
    ρe = -abs.(ρe)
    ρe = ρe / sum(ρe) / dv * C
end

p = ones(2)
ps = Params(p)

rmin = 1e-6
rmax = 2.0
pad = :same
ϕ = Op(r -> 1 / (4π * r), rmax, cell; rmin, pad)
E = Op(r -> 1 / (4π * r^2), rmax, cell; rmin, pad, l=1)
function H(ρe::AbstractArray, ρp::AbstractArray, p)
    ρe = normρe(ρe, dv, C)
    c1, c2 = p
    ϕe = ϕ(ρe)
    ϕp = ϕ(ρp)

    Ve = ϕe ⋅ ρe / 2 * dv
     Vp = ϕp ⋅ ρe * dv
     K = abs(c1) * sum(abs.(ρe) .^ (1 + abs(c2))) * dv

    @show Ve, Vp, K
    @show Ve + Vp + K
end

ρe_ = zeros(size(ρe))
p_ = zeros(size(p))

@show H(ρe, ρp, p)
function loss(ρe, ρp, p)

    # autodiff(x->H(x,ρp,p),Duplicated(ρe,ρe_))
    # l=sum(abs,ρe_)

    l = sum(abs, gradient(x -> H(x, ρp, p), ρe)[1]) / abs(C)
end

@show loss(ρe, ρp, p)
# autodiff(x->loss(ρe,ρp,x),Duplicated(p,p_))
# @show gradient(x->loss(ρe,ρp,x),p)[1]

# plot(ρp)
plot(-ρe)
res = bboptimize(p -> loss(ρe, ρp, p), p; SearchRange=[(0, 2), (0, 2)], MaxFuncEvals=60, TraceMode=:verbose)
@show best_fitness(res)
@show p = best_candidate(res)
@show loss(ρe, ρp, p)

ρe0 = copy(ρe)
data = [()]
opt = ADAM(0.1)
ps = Flux.params(ρe)
Flux.@epochs 50 Flux.train!(() -> H(ρe, ρp, p), ps, data, opt)
ρe = normρe(ρe, dv, C)
@show nae(ρe, ρe0)