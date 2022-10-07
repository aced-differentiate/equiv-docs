d0=d1=d2=0
"""
We demonstrate core capabilities of EquivariantOperators.jl: particle mesh, finite differences, and machine learning. In a particle mesh context, we place 2 +1 point atoms in a grid creating a scalar field. We then do a finite difference calculation of the electric field and potential. Switching gears to machine learning, we train our equivariant neural network to learn this transformation from the charge distribution (scalar field) to the electric potential (scalar field) and field (vector field), essentially learning the Green's function solution to Poisson's Equation. Finally, we treat our point atoms as proton nuclei and train another neural network to predict the ground state electronic density of H2.
"""

repo = ".."
include("$repo/src/neural_networks.jl")
include("$repo/src/atoms.jl")
# include("../src/plotutils.jl")
# include("$repo/src/EquivariantOperators.jl")
# using .EquivariantOperators

using FileIO
using BSON: @save,@load
using IterTools
using Functors

using Plots
using Random
using Optim
using Flux, CUDA

Random.seed!(1)
# ENV["CUDA_VISIBLE_DEVICES"] = -1
mygpu = identity

# name = "AAACQNRQYWKRAR-SCLBCKFNSA-N"
# name = "benzene"
# name = "h2"
# name="dsgdb9nsd_000001"
# case = load("..\\density-prediction\\data\\$name.jld2", "cases")[1]

folder = "..\\density-prediction\\data"
names = [split(x, ".")[1] for x in readdir(folder) if x == "h2.jld2"]
systems = [load("$folder\\$name.jld2", "cases") for name in names]
# file = "$folder\\$name.jld2"
# case = load(file,"case")

data = []
grid=nothing
for (cases, name) in zip(systems, names)
    for case in cases
        @unpack resolution,
        density,
        core_density,
        charges,
        core_charges,
        # atoms,
        # core_atoms,
        positions = case
        atoms=charges

        origin = ones(3)
         cell = resolution*Matrix(I,3,3)
      sz = size(density)
global        grid = Grid(cell, sz; origin)

        ρe = cat(density, dims = 4)
        # ρe *=total/(sum(ρe*dV))

        # x = ρp |> mygpu
        # y = ρe |> mygpu

        push!(data, Frame(atoms,positions,grid;ρe))
    end
end


@unpack ρp,grid,ρe,ops=data[1]
@unpack dv,cell=grid
@unpack ϕ,E=ops
ϕp = ϕ(ρp)
ϕe = ϕ(ρe)
# E=dv*sum(ρe.*ϕp)
# E1=dv*sum(ρe.*ϕe)
##
using GLMakie
GLMakie.inline!(false)
# fig = volume(E.kernel[:, :, :, 1])
# fig = volume(ϕ.kernel[:, :, :, 1])
fig = volume(d4[:, :, :, 1])
display(fig)
#
# ##
# @unpack dx=grid
# rmax=3.
# m=DensityPredictor(dx,rmax)
# ps=Flux.params(m.L,m.D,m.D1)
# # ps=Flux.params(m)
# yhat =0
# function loss(x)
#     y=x.ρe
#     global yhat=m(x)
#     s = sum(y)
#     @show l = nae(yhat, y; s)
#     l
# end
#
# data0 = data
# data = vcat([fill(x, 5) for x in data]...)
# loss(data[1])
# opt = ADAM(0.1)
#
# for i = 1:10
#     Flux.train!(loss, ps, data, opt)
# end
# @save "m.bson" m
#
# ##
# @load "m.bson" m
# function fg!(F, G, x)
#     ρe_hat, ρp = predict_density(x, atoms, grid)
#     ρhat = -ρe_hat + ρp
#     @show forces_hat = calcforces(x, atoms, ρhat, grid)
#     if G != nothing
#         G .= -forces_hat
#     end
#     norm(forces_hat) / sum(atoms)
# end;
#
# x0 = positions
#
# xres = optimize(
#     Optim.only_fg!(fg!),
#     x0,
#     LBFGS(),
#     Optim.Options(show_trace = true, f_tol = tol),
# )
# xmin = Optim.minimizer(xres)
# dmin = norm(xmin[:, 1] - xmin[:, 2])
# @printf "\nOptimal bond length for Ecut=%.2f: %.3f Bohr\n" Ecut dmin
#
# ##
# using GLMakie
# include("$repo/src/plotutils.jl")
#
# # volume(x[:, :, :, 1])
# # volume(y[:, :, :, 1])
# GLMakie.inline!(false)
# # fig=GLMakie.contour(10yhat[:, :, :, 1])
# # fig=volume(10yhat[:, :, :, 1],algorithm = :iso,)
# # fig=volume(10y[:, :, :, 1],algorithm = :iso,)
# # fig=volume(10yhat[:, :, :, 1],algorithm = :absorption,)
# fig = volume(yhat[:, :, :, 1])
# display(fig)
# # volume(10z[:, :, :, 3])
#
# # vis(L)
