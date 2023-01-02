using Functors
using UnPack
using StaticArrays
using NearestNeighbors
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
    Grid(resolutions, origin=ones(length(resolutions)), sz=nothing)
    Grid(resolutions, rmax)
    Grid(cell, origin=ones(size(cell, 1)), sz=nothing)
    Grid(cell, rmax)

Constructs `Grid` for interpolation, coordinate retrieval, and particle mesh placement. Grids can be in any dimension and be boundless or bounded. At minimum, a grid stores its discretization cell and origin in pixels. 
    
For an orthogonal grid, supply a list of resolutions in each dimension. For non-orthogonal grid, supply the cell vectors as a column-wise matrix. Origin by default is at index (1, ...).

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
    - `p`: `Array` of position vectors
    - `r`: `Array` of radial distances (lengths of position vectors)
    - `x`, `y`, `z`: `Array` of that coordinate (field aliases calling `p`)

# Examples
##  2d interpolation
```
dx = dy = 0.1
g = Grid((dx, dy))
a = [x^2 + y^2 for x in 0:dx:0.5, y in 0:dy:0.5]
a[g, 0.25, 0.25]
# 0.13
```
## coordinate retrieval
```
# alternatively the previous example array can be made using a bounded grid and its coordinate fields
R = 0.5
g = Grid((dx, dy), R)
a = g.x.^2 + g.y.^2 # or g.r.^2
```

## grid construction details
```
Grid((0.1,)) # 1d grid spaced 0.1 apart with origin at (1,)
Grid((0.1, 0.2), (2, 5)) # 2d grid spaced 0.1 along x and 0.2 along y, with origin at (2 ,5)
Grid((0.1, 0.1), 20.0) # bounded 2d grid spaced 0.1 apart, halflength 20.0, with origin at (201, 201), pixel size (401, 401)
Grid(0.1 * [1 1 0; 1 0 1; 1 1 1]', ones(3), (10, 12, 15)) # bounded 3d grid with cell vectors [.1, .1, 0], [.1, 0, .1], [.1, .1, .1] (note matrix transpose). origin (1, 1, 1), pixel size (10, 12, 15). can construct lattice this way
```
"""
function Grid(resolutions::Base.AbstractVecOrTuple, origin=ones(length(resolutions)), sz=nothing; kw...)
    Grid(Diagonal(collect(resolutions)), origin, sz; kw...)
end
function Grid(resolutions::Base.AbstractVecOrTuple, rmax::Real; kw...)
    Grid(Diagonal(collect(resolutions)), rmax; kw...)
end

function Grid(cell::AbstractMatrix, rmax::Real; kw...)
    n = size(cell, 1)
    sz = 1 .+ 2 * ceil.(rmax * (cell \ ones(n)))
    Grid(cell, (sz .+ 1) ./ 2, sz; kw...)
end

# With grid info we can interpolate a scalar or vector field at any location. We can also place a scalar or vector point source anywhere with automatic normalization wrt discretization. Both work via a proximity weighted average of the closest grid points (in general up to 4 in 2d and 8 in 3d).
"""
    place!(field, grid, rvec, val)
    
Place a discrete impulse value via particle mesh method, specifically using area weighting according to the [CIC (cloud-in-cell) rule](https://homepage.univie.ac.at/franz.vesely/simsp/dx/node48.html). The impulse is thus spread onto the nearest grid points with proximity weighting and discretization scaling.
    
## particle mesh placement and interpolation
```
dx = dy = dz = 0.1
g = Grid((dx, dy, dz))
a = zeros(5, 5, 5)
v = 1.0
place!(a, g, (0.2, 0.2, 0.2), v)

a[g, 0.2, 0.2, 0.2]
# 1000
v / g.dv
# 1000
```
"""
function place!(field::AbstractArray, grid::Grid, rvec, val)
    for (ix, w) in nearest(grid, rvec)
        field[ix...] += w / grid.dv * val
    end
end
function Base.get(field::AbstractArray, grid::Grid, rvec)
    sum([w * field[ix...] for (ix, w) in nearest(grid, rvec)])
end

function Base.getindex(a::AbstractArray, g::Grid, args...)
    get(a, g, args)
end
function Base.getindex(a::AbstractArray, tree::KDTree, points...)
    a[tree, points]
end
function Base.getindex(a::AbstractArray, tree::KDTree, points::Base.AbstractVecOrTuple)
    k = size(points, 1) + 1
    idxs, dists = knn(tree, points, k)
    w = 1 ./ dists.+1e-12
    w = w / sum(w)
    sum([a[i...] for i in idxs] .* w)
end
function deform(a, v; periodic=false)
    r = zeros(size(a))
    for p in Iterators.product([1:n for n in size(a)]...)
        for (i, w) in nearest(p .+ v[p...])
            if !periodic
                if all(1 .<= i .<= size(a))
                    r[i...] += w * a[p...]
                end
            else
            end
        end
    end
    r
end
# function Base.getindex(a::AbstractArray, i...)
#     a[Grid(ones(ndims(a))),i...,]
# end


function nearest(cell, origin, rvec)
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
    filter(t -> t[2] > 0, res)
end
function nearest(rvec)
    n=length(rvec)
    cell =I(n)
     origin = zeros(n)
    nearest(cell, origin, rvec)
end
function nearest(grid::Grid, rvec)
    @unpack cell, origin = grid
    nearest(cell, origin, rvec)
end


function Base.setindex!(a::AbstractArray, v, g::Grid, p...)
    place!(a, g, p, v)
end

function place!(a::AbstractArray, g1::Grid, g2::Grid, p, b::AbstractArray, ; periodic=false)
    p -= g2.cell * (g2.origin .- 1)
    for (i, w) in nearest(g1, p)
        j = i .+ size(b) .- 1
        ia = max.(i, 1)
        ja = min.(j, size(a))
        ib = 1 .+ ia .- i
        jb = size(b) .+ ja .- j
        if periodic
            l = Iterators.product([zip(sort(unique(vcat([1], reverse(i:-p:1), j+1:p:s, [s]))), sort(unique(vcat([1], reverse(i-1:-p:1), j:p:s, [s])))) for (i, j, p, s) in zip(ib, jb, size(a), size(b))])
            # ir = getindex.(last(l), 1)
            # jr = getindex.(first(l), 2)
            for ijb in l
                ija = map(zip(ijb, ib, jb, ia, ib, size(a))) do ((i, j), ib, jb, ia, ib, p)
                    (ia + mod(i - ib, p), ja + mod(j - jb, p))

                end
                a[[ia:ja for (ia, ja) in ija]...] .+= b[[ib:jb for (ib, jb) in ijb]...] .* w
            end
        else
            a[[ia:ja for (ia, ja) in zip(ia, ja)]...] .+= b[[ib:jb for (ib, jb) in zip(ib, jb)]...] .* w
        end
    end
end
# function Base.place!(f, grid, p::AbstractMatrix, vals)
#     for (val, rvec) in zip(vals, eachcol(p))
#         place!(f, grid, rvec, val)
#     end
# end

function red2cart(x,lattice)
    reduce(hcat,[lattice*x for x in eachcol(x)])
end
function cart2red(x,lattice)
    reduce(hcat,[lattice\x for x in eachcol(x)])
end