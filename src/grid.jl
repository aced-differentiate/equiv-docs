using Functors
using UnPack
using StaticArrays

function harmonics(r::AbstractArray{T}, l) where {T<:Complex}
    θ = l * angle.(r)
    @. complex(cos(θ), sin(θ))
end
function harmonics(r::AbstractArray, l)
    if l == 1
        return r
    end
    dims = length(r[1])
    if dims == 2
        @. r = complex(r[1], r[2])
        return harmonics(r)
    end
end
struct Grid
    cell::AbstractMatrix
    origin::Any
    p
    Y
    r
    dv::Real
end
# struct Grid{T}
#     cell::AbstractMatrix
#     origin::Any
#     r::AbstractArray{T}
#     Y::AbstractArray
#     r::AbstractArray
#     dv::AbstractFloat
# end
@functor Grid

Base.size(g::Grid) = size(g.r)
function Base.getproperty(g::Grid, f::Symbol)
    if f === :x
        return getindex.(g.p, 1)
    elseif f === :y
        return getindex.(g.p, 2)
    elseif f === :z
        return getindex.(g.p, 3)
    end
    getfield(g, f)
end


function Grid(
    cell::AbstractMatrix,
    origin=ones(size(cell, 1)),
    sz=nothing;
    T=SVector
)
    n = size(cell, 1)
    if sz === nothing
        Y = r = p = nothing
    else
        p = [
            T((cell * collect(a .- origin))...)
            for a in Iterators.product([1:a for a in sz]...)
        ]
        r = norm.(p)
        Y = [p ./ (r .+ 1e-16)]
    end
    dv = det(cell)
    cell = SMatrix{n,n}(cell)
    Grid(cell, origin, p, Y, r, dv)
end

"""
    Grid(resolutions, [origin], [sz])
    Grid(resolutions, rmax)
    Grid(cell, [origin], [sz])

Constructs `Grid` struct Used downstream for interpolation (both read/write). Grids can be in any dimension and be Boundless or bounded. At minimum, a grid stores its discretization cell and origin in pixels.  For an orthogonal grid, supply a list of resolutions in each dimension. For non-orthogonal grid, supply the cell vectors as a column-wise matrix. Origin by default is at index (1, ...).

Bounded grids compute and store additional info eg position vectors and their norms of all grid points. To construct bounded grid, supply the overall integer pixel size `sz`. For bounded cubic grid, use convenience constructor with `rmax` 

# Params
- `resolutions`: list of resolutions in each dimension for orthogonal grids
- `origin`: indices of the origin, may be decimal valued
- `sz`: integer size (pixel dimensions) of bounded grid
- `rmax`: halflength (in length units, not pixels) of bounded cubic grid
- `cell`: column-wise matrix of discretization cell vectors

# fields
- `cell`
- `origin`
- additional fields of bounded grids
-- `p`: `Array` of position vectors
-- `r`: `Array` of radial distances (lengths of position vectors)
-- `x`, `y`, `z`: `Array` of that coordinate (field aliases calling `p`)

# Examples
Grid((0.1,)) # 1d grid spaced 0.1 apart with origin at (1,)
Grid((0.1, 0.2), (2, 5)) # 2d grid spaced 0.1 along x and 0.2 along y, with origin at (2 ,5)
Grid((0.1, 0.1), 20.0) # bounded 2d grid spaced 0.1 apart, halflength 20.0, with origin at (201, 201), pixel size (401, 401)
Grid(0.1 * [1 1 0; 1 0 1; 1 1 1]', ones(3), (10, 12, 15)) # bounded 3d grid with cell vectors [.1, .1, 0], [.1, 0, .1], [.1, .1, .1] (note matrix transpose). origin (1, 1, 1), pixel size (10, 12, 15). can construct lattice this way
"""
function Grid(resolutions::Base.AbstractVecOrTuple, origin=ones(length(resolutions)), sz=nothing; kw...)
    Grid(Diagonal(collect(resolutions)), origin, sz; kw...)
end

function Grid(cell::AbstractMatrix, rmax::Real; kw...)
    n = size(cell, 1)
    sz = 1 .+ 2 * ceil.(rmax * (cell \ ones(n)))
    Grid(cell, (sz .+ 1) ./ 2, sz; kw...)
end

"""
    Base.get(field::AbstractArray, grid::Grid, rvec::AbstractVector)
    Base.put!(
        field::AbstractArray,
        grid::Grid,
        rvec::AbstractVector,
        val::AbstractVector,
    )

With grid info we can interpolate a scalar or vector field at any location. We can also place a scalar or vector point source anywhere with automatic normalization wrt discretization. Both work via a proximity weighted average of the closest grid points (in general up to 4 in 2d and 8 in 3d).
"""
function Base.get(field::AbstractArray, grid::Grid, rvec)
    sum([w * field[ix...] for (ix, w) in nearest(grid, rvec)])
end

function Base.getindex(a::AbstractArray, g::Grid, args...)
    get(a, g, args)
end


function nearest(grid, rvec)
    @unpack cell, origin = grid
    n = length(origin)
    # sz = collect(size(grid))
    rvec = collect(rvec)
    ix = origin .+ (cell \ rvec)
    ixfloor = floor.(Int, ix)
    er = ix - ixfloor

    res = [
        (ixfloor .+ p, prod(ones(n) - abs.(p .- er)))
        for
        p in Iterators.product(fill(0:1, n)...)
        #  if ones(n) <= ixfloor .+p<=sz || error("interpolating out of bounds indices")
    ]
        filter(t->t[2]>0,res)
end

function Base.put!(field::AbstractArray, grid::Grid, rvec, val)
    for (ix, w) in nearest(grid, rvec)
        field[ix...] += w / grid.dv * val
    end
end

function Base.setindex!(a::AbstractArray, v, g::Grid, p...)
    put!(a, g, p, v)
end
    
    function Base.put!(a::AbstractArray, b::AbstractArray, g1::Grid,g2::Grid, p)
        p.-=g2.origin.-1
        for (i, w) in nearest(g1,p)
            j=i.+size(b).-1
        a[[i:j for (i,j) in zip(i,j)]...] .+=b.*w
    end
end
# function Base.put!(f, grid, p::AbstractMatrix, vals)
#     for (val, rvec) in zip(vals, eachcol(p))
#         put!(f, grid, rvec, val)
#     end
# end
