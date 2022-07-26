using Random
using Functors
using Zygote

include("grid.jl")
include("conv.jl")
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
Flux.trainable(m::Op) = [m.radfunc]

function makekernel(radfunc, rmin, rmax, l, grid)
    @unpack cell, Y, R, dv, r = grid
    n = size(cell, 1)
    f = r -> rmin <= r <= rmax ? radfunc(r) : 0.0
    rscalars = f.(R)
    if l == 0
        return rscalars * dv
    end

    while l > length(Y)
        push!(Y, harmonics(r, length(Y) + 1))
    end

    kernel = rscalars .* Y[l] * dv
end
function Op(
    radfunc,
    rmax,
    cell::Matrix;
    kw...
)
grid = Grid(cell, rmax)
    Op(radfunc,rmax,grid;kw...)
end
function Op(
    radfunc,
    rmax,
    grid::Grid;
    l = 0,
    rmin=0.,
    convfunc = dspconv,
    pad = :same,
)
    kernel = makekernel(radfunc, rmin, rmax, l, grid)
    convfunc_(x, f; kw...) = convfunc(x, f; pad, kw...)

    Op(l, kernel, grid, convfunc_, radfunc, rmin, rmax)
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
function Op(
    name::Union{Symbol,String},
    cell;
    rmin = 0.0,
    rmax = Inf,
    convfunc = dspconv,
    pad = :same,
    l = 0,
    σ = 1.0,
)
    name = Symbol(name)

    dims = size(cell, 1)
    radfunc = nothing
    if name == :Gaussian
        radfunc = r -> exp(-r^2 / (2 * σ^2)) / sqrt(2π * σ^(2dims))
        rmax = 2σ
        return Op(radfunc, rmin, rmax, cell)
    elseif name in [:△, :▽]
        pad = 0
        l = 1
        grid = Grid(cell, fill(3, dims))


        kernel = [
            sum(abs.(v)) > 1 ? zeros(dims) : -cell' \ collect(v) / 2
            for v in Iterators.product(fill(-1:1, dims)...)
        ]
        if name == :△
            l=0
            kernel = [
                dspconv(kernel, kernel;product=dot)[i...] / 4
                for i in Iterators.product(fill(1:2:5, dims)...)
            ]
        end

        convfunc_(x, f; kw...) = parent(padarray(
            convfunc(x, f; pad, kw...),
            Pad(:replicate, ones(Int, ndims(x))...),
        ))
        return Op(l, kernel, grid, convfunc_, radfunc, rmin, rmax)
    end
    # convfunc_(x, f; kw...) = convfunc(x, f; pad, kw...)
    # Op(l, kernel, grid, convfunc_, radfunc, rmin, rmax)
end

"""
    function (m::Op)(x::AbstractArray, )

"""
function (m::Op)(x::AbstractArray; kw...)
    @unpack grid, kernel, convfunc = m
    convfunc(x, kernel; kw...)
end

function LinearAlgebra.:⋅(m::Op, x)
    m(x; product = ⋅)
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
function nae(yhat, y; sumy = sum(abs.(y)))
    if sumy == 0
        error()
    end
    sum(abs.(yhat .- y)) / sumy
end
