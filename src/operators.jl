using Random
using Functors
# using Zygote

include("grid.jl")
include("conv.jl")
include("diffrules.jl")

include("radfuncs.jl")
# include("diffrules.jl")
Random.seed!(1)

mutable struct Op
    l
    kernel::AbstractArray
    grid::Grid
    convfunc
    radfunc
    rmin
    rmax
end
@functor Op
# Flux.trainable(m::Op) = [m.radfunc]

function makekernel(radfunc, rmin, rmax, l, grid;)
    @unpack cell, Y, r, dv, p = grid
    n = size(cell, 1)
    f = r -> rmin <= r <= rmax ? radfunc(r) : 0.0
    rscalars = f.(r)
    if l == 0
        return rscalars * dv
    end

    while l > length(Y)
        push!(Y, harmonics(p, length(Y) + 1))
    end

    kernel = rscalars .* Y[l] * dv
end

function Op(
    radfunc,
    grid::Grid;
    rmax=Inf,
    l = 0,
    convfunc = dspconv,
    rmin = 0.0,
    pad=:same,
    kw...,
)
    kernel = makekernel(radfunc, rmin, rmax, l, grid)
    convfunc_(x, f; kw1...) = convfunc(x, f;pad, kw..., kw1...)

    Op(l, kernel, grid, convfunc_, radfunc, rmin, rmax)
end
"""
    Op(radfunc, rmax, cell::AbstractMatrix; kw...)

constructs equivariant operator
"""
function Op(radfunc, rmax, cell::AbstractMatrix; cutoff=true,kw...)
    grid = Grid(cell, rmax)
    if !cutoff
        rmax=Inf
    end
    Op(radfunc,  grid; rmax,kw...)
end

"""
    Op(
        name::Union{Symbol,String},
        cell;
        pad =:same,
        rmin = 0,
        rmax = Inf,
        l = 0,
        σ = 1.0,
    )
    Op(
        radfunc,
        rmin::AbstractFloat,
        rmax::AbstractFloat,
        cell;
        l = 0,
        pad =:same,
    )

`Op` constructs finite difference operators. Prebuilt operators like differential operators (`▽`) & common Green's functions can be specified by name. Custom equivariant operators can be made by specifying radial function.
"""

""""
    Del(cell; pad = :same, border = :smooth)

constructs gradient operator
"""
function Del(cell::AbstractMatrix; pad = :same, border = :smooth)
    l = 1
    dims = size(cell,1)
    grid = Grid(cell, fill(2, dims),fill(3, dims))

if dims==1
    kernel=[1,0,-1.]/2/cell[1]
    else
    kernel = [
        SVector{dims}(sum(abs.(v)) > 1 ? zeros(dims) : -cell' \ SVector(v...) / 2)
        for v in Iterators.product(fill(-1:1, dims)...)
    ]
        end

    radfunc = nothing
    rmin = 0.0
    rmax = Inf
    # @show border == :smooth
    if border == :smooth
         function f1(x,f;product=*)
         r=dspconv(x, f; pad=0,product)
         ix=[Int.([1, (1:a)..., a]) for a in size(r)]
            r = r[ix...]
        end
        return Op(l, kernel, grid, f1, radfunc, rmin, rmax)
    else
        convfunc_= (x,f;product=*)->dspconv(x, f; pad, border,product)
        return Op(l, kernel, grid, convfunc_, radfunc, rmin, rmax)
    end
end

"""
    Laplacian(cell; pad = :same, border = :smooth)

constructs Laplacian operator

# Examples
```
```
"""
function Laplacian(cell::AbstractMatrix; pad = :same, border = :smooth)
    l = 0
    dims = size(cell,1)
    grid = Grid(cell, fill(2, dims),fill(3, dims))

    kernel = Del(cell/2).kernel
    kernel = [
        dspconv(kernel, kernel; product = dot)[i...]
        for i in Iterators.product(fill(1:2:5, dims)...)
    ]

    radfunc = nothing
    rmin = 0.0
    rmax = Inf
    if border == :smooth
        function f1(x,f)
        r=dspconv(x, f; pad=0)
        ix=[Int.([1, (1:a)..., a]) for a in size(r)]
           r = r[ix...]
       end
        return Op(l, kernel, grid, f1, radfunc, rmin, rmax)
    else
        convfunc_= (x,f)->dspconv(x, f; pad, border)
        return Op(l, kernel, grid, convfunc_, radfunc, rmin, rmax)
    end
end

"""
    Gaussian(cell, σ, rmax; kw...)

constructs Normalized Gaussian diffusion operator
"""
function Gaussian(cell, σ, rmax; kw...)
    radfunc = r -> exp(-r^2 / (2 * σ^2)) / sqrt(2π * σ^(2ndims(cell)))
    return Op(radfunc, rmax, cell; kw...)
end

# convfunc_(x, f; kw...) = parent(padarray(
#     convfunc(x, f; pad, kw...),
#     Pad(:replicate, ones(Int, ndims(x))...),
# ))



#     end
#     # convfunc_(x, f; kw...) = convfunc(x, f; pad, kw...)
#     # Op(l, kernel, grid, convfunc_, radfunc, rmin, rmax)
# end

"""
    function (m::Op)(x::AbstractArray, )

"""
function (m::Op)(x::AbstractArray; kw...)
    @unpack grid, kernel, convfunc = m
    convfunc(x, kernel; kw...)
end

function LinearAlgebra.:·(m::Op, x)
    m(x; product = ·)
end
function ⨉(m::Op, x)
    m(x; product = ⨉)
end
function remake!(m)
    @unpack radfunc, rmin, rmax, grid, l = m
    m.kernel = makekernel(radfunc, rmin, rmax, l, grid)
end
function Base.abs(x::AbstractArray)
    sum(abs.(x))
end

"""
    function nae(yhat, y; s = sum(abs.(y)))

Normalized absolute error. eg 0.1 denotes 10% absolute error
"""
function nae(yhat, y; s = sum(abs.(y)))
    if s == 0
        error()
    end
    sum(abs.(yhat .- y)) / s
end


function center!(pos)
    n=size(pos,2)
        center=sum(eachcol(pos))/n
        pos.-=repeat(center,1,n)
end

norms(m)=norm.(eachcol(m))

function rescale(a, s)
    sa= sum(a)
    er=(sa-s)/s
    @info "normalization error $er"
        a /sa * s
    end

# function center(a)
#   r=  1/sum(a)*sum(Iterators.product([
#         1:n for n in size(a)
#     ]...)) do ix
#     a[ix...]*collect(ix)
# end
# end

# function center(T,a)
#     T.(center(a))
# end