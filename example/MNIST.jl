"""
Finite difference calculation & machine learning of electric potential & electric field from charge
"""

using LinearAlgebra
using Statistics

using Plots
using Random
using Flux
using MLDatasets

Random.seed!(1)
include("../src/operators.jl")
include("../src/plotutils.jl")
# using Main.EquivariantOperators # end users should omit Main.

train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

n=1024
train_x=train_x[:,:,1:n]
train_y=train_y[1:n]

sz=size(train_x)[1:2]
cell=[1 0; 0 1]/sz[1]
grid=Grid(cell,sz;T=complex)
ops=[Op(r->1, .5,grid;l,convfunc=cvconv) for l=0:4]

function feat(x)
    r=[op(x)[1] for op in ops]
    vcat([r[1]],[[real(z),imag(z)] for z in r[2:end]]...)
end

X=1e3*hcat(feat.(eachslice(train_x;dims=3))...)
Y=train_y

n=size(X,1)
σ=leakyrelu
nn=Chain(Dense(n,4n,σ),Dense(4n,10),softmax)
# nn=Chain(Dense(n,2n,σ),Dense(2n,4n,σ),Dense(4n,10),softmax)

function loss(x,y)
    x=eachcol(x)
    # l= @. nn(x)[y+1]
    @show l=mean([-log(nn(x)[y+1]) for (x,y) in zip(x,y)])
end

# data =zip(eachcol(X),train_y)
data = Flux.Data.DataLoader((X, Y), batchsize=32)
loss(first(data)...)
opt = ADAM(0.1)
ps=Flux.params(nn)
Flux.@epochs 10 Flux.train!(loss, ps, data, opt)
