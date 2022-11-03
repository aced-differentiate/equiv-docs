using LinearAlgebra
using Statistics

using Plots
using Random
using Flux
using MLDatasets
using Images
using LegendrePolynomials

Random.seed!(1)
include("../src/operators.jl")
include("../src/plotutils.jl")
# using Main.EquivariantOperators # end users should omit Main.

data=MNIST
# data=CIFAR10
nclasses=10
batchsize=128
# nsamples = length(data(:train))
# nsamples = 2000
# nepochs=8000÷(nsamples÷batchsize)

# s= :
nepochs=800

train_x, train_y =data.traindata()
test_x, test_y = data.testdata()
sz = size(train_x)[1:2]

# train_x =train_x[fill(:,ndims(train_x)-1)...,1:nsamples]
# train_y = train_y[1:nsamples]

##
dx=1/sz[1]
cell = [1 0; 0 1] *dx

grid = Grid(cell, sz; T=complex)
convfunc(x,f)=overlapdot(x,conj.(f);)
ops1 = [Op(r -> Pl( 1.5r,k), grid; l, rmax=.6, convfunc) for l = 0:6, k = 0:2]
ops2 = [Op(r ->exp(-((r-c)/.1)^2), grid; l, rmax=.5, convfunc) for l = 0:6, c=0:.1:.4]
ops=hcat(ops1,ops2)

pad = :same
border = :smooth
▽ = Del(cell; pad, border)

function feat(a)
 r=[op(a) for op in ops]
 vcat([[real(z),imag(z)] for z in r]...)
end

X = hcat(feat.(eachslice(train_x, dims=ndims(train_x)))...)
Y = train_y

n = size(X, 1)
nn = Chain(Dense(n, 4nclasses, leakyrelu),Dense(4nclasses, 2nclasses, leakyrelu), Dense(2nclasses, nclasses))
# nn = Chain(Dense(n, n, leakyrelu),Dense(n, 2n, leakyrelu), Dense(2n, nclasses), softmax)
# nn=Chain(Dense(n,2n,leakyrelu),Dense(2n,4n,leakyrelu),Dense(4n,10),softmax)
ps = Flux.params(nn)

function loss(x, y)
    # l= @. nn(x)[y+1]
    res=hcat(nn.(eachcol(x))...)
    @show accuracy=mean(argmax.(eachcol(res)).==(y.+1))
    # @show l = mean([-log(p[y+1]) for (p, y) in zip(res, y)])
    y=Flux.onehotbatch(y,0:nclasses-1)
    @show l = Flux.logitcrossentropy(res,y)
    # @show penalty= .0001sum(abs, ps)
    # l+penalty
end

# data =zip(eachcol(X),train_y)
data = Flux.Data.DataLoader((X, Y); batchsize)
loss(first(data)...)
opt = ADAM(0.1)
Flux.@epochs nepochs Flux.train!(loss, ps, data, opt)

# heatmap(reverse(train_x[:,:,1];dims=1)')
loss(X,Y)
Xtest = hcat(feat.(eachslice(test_x, dims=ndims(train_x)))...)
loss(Xtest,test_y)
