using Functors
using UnPack
using StaticArrays

function harmonics(r::AbstractArray{T},l) where T<:Complex
 θ=l*angle.(r)
    @. complex(cos(θ),sin(θ))
end
function harmonics(r::AbstractArray,l)
    if l==1
        return r
    end
    dims=length(r[1])
    if dims==2
@. r=complex(r[1],r[2])
return harmonics(r)
end
end
struct Grid{T}
    cell::AbstractMatrix
    origin::Any
    r::AbstractArray{T}
    Y::AbstractArray
    R::AbstractArray
    dv::AbstractFloat
end
@functor Grid

Base.size(g::Grid) = size(g.r)
function Base.getproperty(g::Grid, f::Symbol)
    if f === :x
        return getindex.(g.r, 1)
    elseif f === :y
        return getindex.(g.r, 2)
    elseif f === :z
        return getindex.(g.r, 3)
    end
    getfield(g, f)
end



"""
    Grid(cell, rmax::AbstractFloat)
    Grid(
        cell::AbstractMatrix,
        sz::Union{AbstractVector,Tuple};
        origin = (sz .+ 1) ./ 2,
        )

Grid is specified by its discrete cell vectors (column-wise matrix), overall size and origin. For a uniform Cartesian 5x5 grid discretized at 0.1 with a centered origin, we get `cell = [0.1 0; 0 0.1]` & `origin = [3, 3]`. Grid cell can in general be noncartesian.
"""
function Grid(
    cell::AbstractMatrix,
    sz::Union{AbstractVector,Tuple};
    origin = (sz .+ 1) ./ 2,
    T=SVector
)
    n = size(cell, 1)
        rvecs = [
T(            (cell * collect(a .- origin))...)
            for a in Iterators.product([1:a for a in sz]...)
        ]
    R = norm.(rvecs)
    Y =[ rvecs ./ (R .+ 1e-16)]
    dv = det(cell)
    cell=SMatrix{n,n}(cell)
    Grid(cell, origin, rvecs, Y, R, dv)
end

function Grid(cell, rmax::Real)
    n = size(cell, 1)
    Grid(cell, 1 .+ 2 * ceil.(rmax * (cell \ ones(n))))
end


struct Props

    nranks::Any
    ranks::Any
    parities::Any
    slices::Any
end

function Props(ranks)
    Props(; ranks)
end
function Props(; nranks = nothing, ranks = nothing)
    if ranks === nothing
        ranks = vcat([fill(i - 1, n) for (i, n) in enumerate(nranks)]...)
    elseif nranks === nothing
        nranks = zeros(Int, maximum(ranks) + 1)
        for x in ranks
            nranks[x+1] += 1
        end
    end
    ends = Int.(cumsum(2 .* ranks .+ 1))
    slices = [(i == 1 ? 1 : ends[i-1] + 1):ends[i] for i in eachindex(ends)]

    # starts = [1, (1 .+ cumsum(nranks))...]
    # grouped_slices = [slices[starts[i]:starts[i+1]-1] for i = 1:length(nranks)]
    parities = (-1) .^ ranks
    Props(nranks, ranks, parities, slices)
end

"""
    (field::AbstractArray)(rvec, grid::Grid)

Makes field objects callable. Returns interpolated tensor value of the tensor field at position rvec.
"""
function Base.get(fields::AbstractArray, props::Props, i::Int)
    fields[:, :, :, props.slices[i]]
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
function Base.get(field::AbstractArray, grid::Grid, rvec::AbstractVector)
    sum([w * field[ix...] for (ix, w) in nearest(grid, rvec)])
end
function Base.get(
    fields::AbstractArray,
    props::Props,
    i::Int,
    rvec::AbstractVector,
)
    field = get(fields, props, i)
    get(field, props.grid, rvec)
end
function Base.get(fields::AbstractArray, props::Props, rvec::AbstractVector)
    get(fields, props, 1, rvec)
end


function nearest(grid, rvec)
    @unpack cell, origin = grid
    n = length(size(grid))
    ix = origin .+ (cell \ rvec)
    ixfloor = floor.(Int, ix)
    er = ix - ixfloor

    if n == 2
        res = [
            (ixfloor + [x, y], prod(ones(n) - abs.([x, y] - er)))
            for
            x = 0:1, y = 0:1 if [0, 0] < ixfloor + [x, y] < collect(size(grid))
        ]
    elseif n == 3
        res = [
            (ixfloor + [x, y, z], prod(ones(3) - abs.([x, y, z] - er)))
            for
            x = 0:1,
            y = 0:1,
            z = 0:1 if [0, 0, 0] < ixfloor + [x, y, z] < collect(size(grid))
        ]
    end
    res
end

function Base.put!(field::AbstractArray, grid::Grid, rvec::AbstractVector, val)
    for (ix, w) in nearest(grid, rvec)
        field[ix...] += w / grid.dv * val
    end
end

function Base.put!(f, grid, rvecs::AbstractMatrix, vals)
    for (val, rvec) in zip(vals, eachcol(rvecs))
        put!(f, grid, rvec, val)
    end
end
