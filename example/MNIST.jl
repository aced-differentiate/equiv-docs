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
sz=size(train_x)[1:2]

n=1024
train_x=train_x[:,:,1:n]
train_x=reverse.(eachslice(train_x;dims=3);dims=2)
train_y=train_y[1:n]

cell=[1 0; 0 1]/sz[1]
grid=Grid(cell,sz;T=complex)
ops=[Op(r->1, .5,grid;l,convfunc=cvconv) for l=0:4]

function feat(x)
    r=[op(x)[1] for op in ops]
    vcat([r[1]],[[real(z),imag(z)] for z in r[2:end]]...)
end

X=hcat(feat.(train_x)...)
Y=train_y

# ##
# y=8
# # heatmap(train_x[:,:,[Y.==8][1]]')
# # heatmap(real.(ops[2].kernel)')
# plot(heatmap(imag.(ops[2].kernel)'))
# # plot([heatmap(train_x[Y.==y][i]') for i =1:16]...,layout=(4,4))
# Xy=X[:,Y.==y]
# train_xy=train_x[Y.==y]
##
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
Flux.@epochs 50 Flux.train!(loss, ps, data, opt)

# heatmap(reverse(train_x[:,:,1];dims=1)')
