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
nsamples = length(data(:train))
# nsamples = 2000
# nepochs=8000÷(nsamples÷batchsize)
nepochs=1000

train_x, train_y =data.traindata()
test_x, test_y = data.testdata()
sz = size(train_x)[1:2]

train_x =train_x[fill(:,ndims(train_x)-1)...,1:nsamples]
train_y = train_y[1:nsamples]
##
dx=1/sz[1]
cell = [1 0; 0 1] *dx

grid = Grid(cell, sz; T=complex)
convfunc(x,f)=overlapdot(x,conj.(f);)
ops1 = [Op(r -> Pl( 1.5r,k), grid; l, rmax=.6, convfunc) for l = 0:6, k = 0:2]
ops2 = [Op(r ->exp(-((r-c)/.1)^2), grid; l, rmax=.5, convfunc) for l = 0:6, c=0:.1:.4]
ops=hcat(ops1,ops2)

# rmaxs=.5:-.1:.2
# # rmaxs=[.5]
# grids =[
#      Grid(cell,rmax; T=complex) for rmax in rmaxs]
# convfuncs=[
#     (x,f)->cvconv(x,f;pad=(.5rmax)÷dx,stride=min(1,rmax÷dx÷5)) for rmax in rmaxs
#     ]
# radfuncs=[r->exp2(-r/σ) for σ in rmaxs]

# radfuncs=[r->exp2(-r/σ) for σ in rmaxs]
# radfuncs=[r->1]
# radfuncs=[r->1,r->r]
# convfunc(x,f)=overlapdot(x,conj.(f);center=center(round,x))
# convfunc(x,f)=overlapdot(x,conj.(f))
# ops = [Op(r -> cos(k * π * r), grid; l, rmax=Inf, convfunc) for l = 0:10, k = 0:1]
# ops = [Op(radfunc, grid; l, rmax, convfunc) for l = 0:8, (radfunc,grid,rmax,convfunc) in zip(radfuncs,grids,rmaxs,convfuncs)]
a=ops[2](train_x[:,:,1])
maximum(a)
pad = :same
border = :smooth
▽ = Del(cell; pad, border)

function invariants(a)
    norms = abs.(a)
    # phases=a*conj(a[2,1])
    # a0 = pop!(a)
    phases = angle.(a)
    # phases =vcat([phases[i,:]/(i-1)-phases[i-1,:]/(i-2) for i in 3:size(phases,1)]...)
    phases1 =vcat([(i-2)*phases[i,:]-(i-1)*phases[i-1,:] for i in 3:size(phases,1)]...)
    phases2 =vcat([phases[:,i]-phases[:,i+1] for i in 1:size(phases,2)-1]...)
    # vcat(vec.([norms, cos.(phases), ])...)
    # vcat(vec.([norms, cos.(phases), sin.(phases)])...)
    vcat(vec.([norms, cos.(phases1), sin.(phases1), cos.(phases2), sin.(phases2)])...)
    # r = vec(norms)
end

g = e = c = img = 0
function feat(x)
    if ndims(x)==3
    c = colorview(RGB, eachslice(x;dims=3)...)
    g = float.(Gray.(c))

    global g = float.(channelview(g))
    global img = norm.(▽(g))
    else
    global img = x
    end

    r=[op(img) for op in ops]
    # ixs=argmax.(r[1,:])
    # r=hcat([[a[ix] for a in col] for (col ,ix) in zip(eachcol(r),ixs)]...)
    # invariants(maximum.([op(img) for op in ops]))
    # invariants([op(img) for op in ops])

    invariants(r)
end

X = hcat(feat.(eachslice(train_x, dims=ndims(train_x)))...)
Y = train_y

# ##
# y=8
# # heatmap(train_x[:,:,[Y.==8][1]]')
# heatmap(real.(ops[2].kernel)')
# plot(heatmap(imag.(ops[2].kernel)'))
##
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
