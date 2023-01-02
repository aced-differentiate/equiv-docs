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
    f = r -> rmin-1e-16 <= r <= rmax+1e-16 ? radfunc(r) : 0.0
rscalars = f.(r)
    if l == 0
        return rscalars
    end

    while l > length(Y)
        push!(Y, harmonics(p, length(Y) + 1))
    end

    kernel = rscalars .* Y[l] 
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

"""
    Op(radfunc, rmax, resolutions; l = 0, rmin = 0., pad = :same)
    Op(radfunc, rmax, cell; kw...)
    Op(radfunc, grid; kw...)

`Op` constructs equivariant finite difference operators & custom Green's functions by specifying the radial function of the impulse response. Prebuilt operators like differential operators (`▽`) & common Green's functions can be constructed instead using `Del`, `Lap`.

# Args
- `radfunc`: radial function

# Keywords
- `l`: rotation order, `0` for scalar field, `1` for vector field

# Example
```
dx = dy = 0.1
resolutions = (dx, dy)
rmin = 1e-9
rmax = 0.2
ϕ = Op(r -> 1 / r, rmax, resolutions; rmin) # 1/r potential
F = Op(r -> 1 / r^2, rmax, resolutions; rmin, l=1) # 1/r^2 field

g = Grid(resolutions,)
a = zeros(5, 5)
a[3, 3] = 1.0 / g.dv # puts discrete value integrating to 1.0 onto array

ϕ(a)
#=
5×5 Matrix{Float64}:
 0.0   0.0       5.0   0.0      0.0
 0.0   7.07107  10.0   7.07107  0.0
 5.0  10.0       0.0  10.0      5.0
 0.0   7.07107  10.0   7.07107  0.0
 0.0   0.0       5.0   0.0      0.0
=#

F(a)
#=
5×5 Matrix{SVector{2, Float64}}:
 [0.0, 0.0]    [0.0, 0.0]            [-25.0, 0.0]   [0.0, 0.0]           [0.0, 0.0]
 [0.0, 0.0]    [-35.3553, -35.3553]  [-100.0, 0.0]  [-35.3553, 35.3553]  [0.0, 0.0]
 [0.0, -25.0]  [0.0, -100.0]         [0.0, 0.0]     [0.0, 100.0]         [0.0, 25.0]
 [0.0, 0.0]    [35.3553, -35.3553]   [100.0, 0.0]   [35.3553, 35.3553]   [0.0, 0.0]
 [0.0, 0.0]    [0.0, 0.0]            [25.0, 0.0]    [0.0, 0.0]           [0.0, 0.0]
=#
```
"""
function Op(radfunc, rmax, a; cutoff=true,kw...)
    grid = Grid(a, rmax)
    if !cutoff
        rmax=Inf
    end
    Op(radfunc,  grid; rmax,kw...)
end


"""
    Del(resolutions; pad = :same, border = :smooth)
    Del(cell; pad = :same, border = :smooth)

constructs gradient operator (also divergence, curl) using central difference stencil. By default, boundaries are smooth (C1 or C2 continuous) and output is of same length as input.

# Example
## 1d derivative
```
dx = 0.1
x = 0:dx:.5
y = x .^ 2
d = Del((dx,))
d(y)

#=
6-element Vector{Float64}:
 0.0
 0.2
 0.4
 0.6
 0.8
 1.0
=#
```

## 2d gradient
```
dy = dx = 0.1
a = [x^2 + y^2 for x in 0:dx:0.5, y in 0:dy:0.5]
▽ = Del((dx, dy))
grad_a = ▽(a)

#=
6×6 Matrix{SVector{2, Float64}}:
[0.0, 0.0]  [0.0, 0.2]  [0.2, 0.4]  [0.2, 0.6]  [0.2, 0.8]  [0.2, 0.8]
 [0.2, 0.0]  [0.2, 0.2]  [0.2, 0.4]  [0.2, 0.6]  [0.2, 0.8]  [0.2, 0.8]
 [0.4, 0.2]  [0.4, 0.2]  [0.4, 0.4]  [0.4, 0.6]  [0.4, 0.8]  [0.4, 0.8]
 [0.6, 0.2]  [0.6, 0.2]  [0.6, 0.4]  [0.6, 0.6]  [0.6, 0.8]  [0.6, 0.8]
 [0.8, 0.2]  [0.8, 0.2]  [0.8, 0.4]  [0.8, 0.6]  [0.8, 0.8]  [0.8, 1.0]
 [0.8, 0.2]  [0.8, 0.2]  [0.8, 0.4]  [0.8, 0.6]  [1.0, 0.8]  [1.0, 1.0]
=#
``` 
## 2d divergence
```
dy = dx = 0.1
a = [[2x,3y] for x in 0:dx:0.5, y in 0:dy:0.5]
▽ = Del((dx, dy))
▽ ⋅ (a)
#=
6×6 Matrix{Float64}:
 5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0
=#
```
"""
function Del(a; pad = :same, border = :smooth)
        cell=Grid(a).cell
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
        kernel/=grid.dv

    radfunc = nothing
    rmin = 0.0
    rmax = Inf
    # @show border == :smooth
        convfunc_= (x,f;product=*)->dspconv(x, f; pad, border,product)
        return Op(l, kernel, grid, convfunc_, radfunc, rmin, rmax)
end

"""
    Lap(cell; pad = :same, border = :smooth)

constructs Laplacian operator

# Examples
```
# 2d Example
dy = dx = 0.1
a = [x^2 + y^2 for x in 0:dx:0.5, y in 0:dy:0.5]
▽2 = Lap((dx, dy))
▽2(a)

#=
6×6 Matrix{Float64}:
 4.0  4.0  4.0  4.0  4.0  4.0
 4.0  4.0  4.0  4.0  4.0  4.0
 4.0  4.0  4.0  4.0  4.0  4.0
 4.0  4.0  4.0  4.0  4.0  4.0
 4.0  4.0  4.0  4.0  4.0  4.0
 4.0  4.0  4.0  4.0  4.0  4.0
=#
```
"""
function Lap(a; pad = :same, border = :smooth)
    l = 0
    dims = size(a,1)
    grid = Grid(a./2, fill(2, dims),fill(3, dims))
    
    kernel = Del(grid.cell).kernel*grid.dv
    kernel = [
        dspconv(kernel, kernel; product = dot)[i...]
        for i in Iterators.product(fill(1:2:5, dims)...)
            ]/grid.dv/4
            
            grid = Grid(a, fill(2, dims),fill(3, dims))
    radfunc = nothing
    rmin = 0.0
    rmax = Inf
    convfunc_= (x,f;product=*)->dspconv(x, f; pad, border,product)
    return Op(l, kernel, grid, convfunc_, radfunc, rmin, rmax)
end

"""
    Gauss(resolutions, σ, rmax; kw...)
    Gauss(cell, σ, rmax; kw...)

constructs Gaussian diffusion operator with volume Normalized to 1 wrt grid support
"""
function Gauss(a, σ; rmax=2σ, kw...)
    cell=Grid(a).cell
    radfunc = r -> exp(-r^2 / (2 * σ^2)) / sqrt((2π * σ^2)^size(cell,1))
    return Op(radfunc, rmax, cell; kw...)
end

function Ewald(a,sz; kw...)
    cell=Grid(a).cell
    n=size(cell,1)
    g=Grid(cell,ones(n),sz)
    kernel=zeros(sz)
    for i in Iterators.product(fill(-1:0 ,n)...)
        v=sum(i.*eachcol(cell))
        kernel+=map(g.p) do x 
            y=norm(x+v)
            y==0 ? 0 : 1/y
        end
    end
    kernel=fft(kernel)
    cf=(x,f)->fft(x).*f|>ifft|>real
    radfunc= rmin= rmax=nothing
    l=0
    Op(l, kernel, grid, cf, radfunc, rmin, rmax)
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
    convfunc(x, kernel; kw...)*grid.dv
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
#     using Interpolations
#     using DataStructures
#     function getr(a,c)
#         g=Grid(c,ones(ndims(a)),size(a))
#         d=SortedDict([k=>v for (k,v) in zip(g.r,a)])
#         it=LinearInterpolation(collect(keys(d)),collect(values(d)))
#         # it=CubicSplineInterpolation(vec(g.r),vec(a))
#         dr=g.dv^(1/ndims(a))/2
#         return [it[r] for r =0:dr:maximum(g.r)],dr
#     end
    
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
