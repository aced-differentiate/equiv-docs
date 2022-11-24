"""
We demonstrate core capabilities of EquivariantOperators.jl: particle mesh, finite differences, and machine learning. In a particle mesh context, we place 2 +1 point Z in a grid creating a scalar field. We then do a finite difference calculation of the electric field and potential. Switching gears to machine learning, we train our equivariant neural network to learn this transformation from the charge distribution (scalar field) to the electric potential (scalar field) and field (vector field), essentially learning the Float32reen's function solution to Poisson's Equation. Finally, we treat our point Z as proton nuclei and train another neural network to predict the ground state electronic density of H2.
"""

include("../src/operators.jl")
using Flux
using Zygote
# using Enzyme
using BlackBoxOptim
using Images
using Unitful

using FileIO
using GLMakie
using Random
using StatsBase
using PeriodicTable
Random.seed!(1)

name = "small_molecules_v1"
# name = "qm9_20"

EN=Dict([
    1=>2.2,
6=>2.55,
7=>3.04,
8=>3.44,
9=>3.98,
14=>1.9,
16=>2.58,
17=>3.16,
])
dir="..\\data"
# dir="..\\density-prediction\\data"
cases = load("$dir\\$name.jld2", "cases")

# ρe1 = -abs.(cases[1].density)[:,:,:,1]
# y_ = -abs.(cases[end].density)[:,:,:,1]
# y = imresize(ρe1,size(y_))
# @show nae(y,y_)

function mix!(X)
    n = length(X)
    for i = 1:n
        for j = i:n
            push!(X, X[i] .* X[j])
        end
    end
end

Base.Float32(a::AbstractArray)=Float32.(a)
# Float32(x::Real)=convert(Float32,x)

data = []
data_ = []
# for case in cases[[3]]
# for case in cases[1:12]
    for case in cases
    # case = cases[1]
    # global ρv,ρv_SAD, ρc_SAD, ρ, ρp, ρpc, ρpv, ρmh, ρen, ops, X0, X0_, X
    @unpack positions, Z, Zv, Zc, dx, ρv,ρv_SAD, ρc_SAD, ρ = case
    y = ρv
    # y = abs.(ρv+ρc_SAD)

    # volume(abs.(y))
    # volume(abs.(ρp))

    sz = size(y)
    cell = dx * Matrix(I, 3, 3)
    origin = ones(3)
    grid = Grid(cell, sz; origin)
    @unpack dv = grid

    N = sum(Z)
    Nv = sum(Zv)
    Nc = sum(Zc)
    @assert N==Nv+Nc
    @show sum(y) * dv, Nv
    @show sum(ρc_SAD)*dx^3,Nc

    ρpv = zeros(sz)
    place!(ρpv, grid, positions, Zv)
    ρpc = zeros(sz)
    place!(ρpc, grid, positions, Zc)
    ρp = ρpc + ρpv

    # ρmh = zeros(sz)
    # place!(ρmh, grid, positions, [ustrip(elements[z].molar_heat) for z in Z])
    ρen = zeros(sz)
    place!(ρen, grid, positions, [EN[z] for z in Z])

    function rescale(a, s)
        a / sum(a) * s
    end

    # rmax = 3.0
    pad = :same
    alg = :fft
    s = [0.5, 1]
    # s = [0.5, 1, 1.5]
    # s = [1, 2]
    ops = vcat(
        [Op(r -> exp(-r / a), 6a, cell; pad, alg) for a = s],
        [Op(r -> exp(-(r / a)^2), 3a, cell; pad, alg) for a = s],
        [Op(r -> 1 / r, 6.0, cell; pad, alg, rmin=1e-6)],
    )

    #  [op(ρpv) for op in ops],
    #  f_ρpc=[op(ρpc) for op in ops]
    #  f_ρc_SAD=[op(ρc_SAD) for op in ops]
    #  f_ρv=[op(ρv) for op in ops]

    X0 = [vec([f(u) for f = ops, u = (ρp,  ρpv, ρen,ρc_SAD,ρv_SAD)])..., ρc_SAD,ρv_SAD]
    # X0 = [vec([f(u) for f = ops, u = (ρp,  ρpv, ρen)])..., ρc_SAD,ρv_SAD]
    # X0_ = copy(X0)
    mix!(X0)
    # append!(X0,[x.^2 for x in X0])

    # mix!(X)
    # X = [X0..., [op(ρv) for op in ops[[]]]...]
    
    # X0=Float32(X0)
    # X=Float32(X)
    
    # ops = vcat(
    #     [Op(r -> exp(-r / a), 4a, cell; pad, alg,l=1) for a = [0.5, 1, 1.5]],
    # )
    
    # X1 = [op(ρp) for op in ops]
    # n = length(X1)
    # for i = 1:n
    #     for j = i:n
    #         push!(X, X1[i] .⋅ X1[j])
    #     end
    # end

    # volume(X[1])

    # n=5000N
    n=length(ρ)÷4
    Random.seed!(1)
    ix=sample(1:length(ρ),n)
    # ix=sample(collect(1:length(ρ)),weights(vec(ρ)),n)
    
    # ix = eachindex(ρ)
    # ix=1:4:length(ρ)

    # @time    X = collect.(zip(X...))
    # @time A = hcat(X...)'
    # b = vec(y)

    A0 = [X0[i][j] for i in eachindex(X0), j in ix]
    b = y[ix]
    
    
    # push!(data, (A[ix,:], b[ix]))
    
    # push!(data_, (A, b))
    
    # X = [X0..., [op(ρv) for op in ops]...]
    # A = [X[i][j] for i in eachindex(X), j in ix]
    # t=(;dx,A0,A, X0, b,ρc_SAD,ρv)

    t=(;dx,A0, X0, b,ρv_SAD,ρc_SAD,ρv)
    # t=NamedTuple([k=>Float32(v) for (k,v) in pairs(t)])
    push!(data, t)
end

# indexin([missing],A0)
# all(isfinite, A0)
# for x in [ρv, ρc_SAD, ρ, ρp, ρpc, ρpv, ρmh, ρen, ops, X0, X0_, X]
# @show any(isnan, x)
# end
# findall(isnan,A0)

# data=NamedTuple.(data)

data_train=data
# data_train=data[1:8]
A0 = hcat(getproperty.(data_train, :A0)...)
# A = hcat(getproperty.(data_train, :A)...)
b = vcat(getproperty.(data_train, :b)...)
# ρc_SAD= getindex.(data, 3)
# X0= getindex.(data, 4)
# ρc_SAD= getindex.(data, 6)

@show size(A0)
# i=1:4:length(b)
# p0 = A0'[i,:] \ b[i]
p0 = A0' \ b
# @time p0 = A0' \ b
# @time p0 = qr(A0') \ b
#  p0 = pinv(A0') * b
bhat0 = A0' * p0
# bhat0 = max.(0, A0' * p0)
# @show nae(A*p, b)
@show nae(bhat0, b)

# bhat = max.(0, A' * p)
# p = A' \ b
# bhat = A' * p
# @show nae(bhat, b)

# for (ρc_SAD,X0,ρv) in zip(ρc_SAD,X0,ρv)
for t in data[1:1]
    @unpack ρc_SAD,X0,ρv,dx=t
     ρv0=sum(p0.*X0)
     @show nae(ρv0,ρv)
     #  ρv0=dot(p0,X0)
     ρv0=max.(0,ρv0-.8ρc_SAD)
     @show nae(ρv0,ρv)

     
     #  X =[X0...,[op(ρv0) for op in ops[[2,7]]]...]
    #  ρv1=sum(p.*X)*dx^3
    #  @show nae(ρv1,ρv)
    #  ρv1=max.(0,ρv1-.8ρc_SAD)
    #  @show nae(ρv1,ρv)
    
    println()
end

using Plots:plot
plot(ρc_SAD)

plot(ρv)

# volume(ρc_SAD)
# volume(ρv)
     
# ρv0 = map(u->max.(0, dot(u, p0)),zip(X0...))
# ρv0 = map(i->max.(0, dot(getindex.(X0,i), p0)),eachindex(view(X0[1])))
# # X =[vec([f(u) for f in ops, u=[ρc_SAD,ρpc,ρv0]]    )...,ρc_SAD]
# mix!(X)
# ρv1 = map(u->max.(0, dot(u, p)),zip(X...))

# A = vcat(getindex.(data_, 1)...)
# b = vcat(getindex.(data_, 2)...)
# bhat = max.(0, A * p)
# @show nae(bhat, b)
# bhat=reshape(bhat,sz)
# volume(bhat)
# volume(y)

# n =size(A,1)
# # nn=Chain(Dense(n,1))
# nn = Chain(Dense(n, 2n, leakyrelu), Dense(2n, 1))
# ps = Flux.params(nn)
# f_(x) = nn(x)[1]
# f(x)=abs(f_(x)-f_(zeros(n)))

# bhat = 0
# s=sum(b)
# function loss()
#     bhat = f.(eachcol(A))
#     global bhat =rescale(bhat,s)

#     # volume(ρp)
#     # volume(bhat)
#     # volume(-y)
#     # volume(-y_)
#     @show nae(bhat,b)
# end

# data = [()]
# opt = ADAM(0.1)
# Flux.@epochs 200 Flux.train!(loss, ps, data, opt)
