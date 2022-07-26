# using Flux: conv
using LinearAlgebra
using ImageFiltering

function FieldProd(l1, l2, l)
    # l1, l2, l = ranks
    if l1 == 0
        return prod0__
    elseif l2 == 0
        return prod_0_
    elseif l1 == l2 && l == 0
        return prod__0
    elseif l1 == l2 == l == 1
        return prod111
    end

    # (x, y) -> yield(f(x, y), x.grid)
    # (x,y)->yield
end


function getl(x)
    n = size(x)[end]
    if n == 1
        return 0
    end

    dims = ndims(x) - 1
    if dims == 2
        return 1
    elseif dims == 3
        return (n - 1) ÷ 2
    end

end
function tp(x, f, l, p)
    r = l .+ size(f) .- 1
    xl = max.(l, 1)
    xr = min.(r, size(x))
    fl = 1 .+ xl .- l
    fr = size(f) .- r .+ xr
    sum(p.(
        x[[a:b for (a, b) in zip(xl, xr)]...],
        f[[a:b for (a, b) in zip(fl, fr)]...],
    ))
end
"""
    cvconv(x, f; product = *, stride = 1, pad = 0)

"convolution" in computer vision for any dimension, same as Cross correlation. For convolution in signal processing , use `dspconv` instead.

`x` input array
`f` filter array
`product` product in convolution, eg `*`, `dot`
`pad` amount of padding or padding option
- any integer:
- `:same`: adds enough padding so output is same size as input
- `:outer`: output size is `size(x) .+ size(f) .- 1`
`border` type of padding
- 0
- `:replicate` repeats edge values
- `:circular` periodic BC
- `:smooth` continuous derivatives at boundaries useful for differential operators
- `:reflect` reflects interior across boundaries which are not repeated
- `:symmetric` same as `:reflect` but with boundaries repeated
"""
function cvconv(x, f; product = *, stride = 1, pad = 0, border = 0)
    if pad == :outer
        pad = size(f) .- 1
    elseif pad == :same
        pad = (size(f) .- 1) .÷ 2
    elseif length(pad) == 1
        pad = fill(pad[1], ndims(x))
    end
    if border == 0
        l = Iterators.product([
            a:stride:b
            for
            (a, b) in
            zip(ones(Int, ndims(x)) .- pad, size(x) .- size(f) .+ 1 .+ pad)
        ]...)
        return [tp(x, f, l, product) for l in l]
    else
        # if border==:circular
        x = parent(padarray(x, Pad(border, pad...)))
        l = Iterators.product([
            1:stride:b for b in size(x) .- size(f) .+ 1 .+ pad
        ]...)
        return [
            sum(product.(
                x[[a:b for (a, b) in zip(l, l .+ size(f) .- 1)]...],
                f,
            )) for l in l
        ]

    end
end

function cvconv(x::AbstractArray{T}, f::AbstractArray{T}; kw...) where T<:Complex
    cvconv(x,f;product=(x,y)->x*conj(y),kw...)
end
"""
    dspconv(x, f; product = *,pad = :outer,border=0)

Convolution in signal processing. For "convolution" in computer vision, use cvconv instead. By default output size is `size(x) .+ size(f) .- 1`. See `cvconv` for its keyword options which also apply here
"""
function dspconv(x, f; product = *, pad = :outer, border = 0)
    cvconv(x, reverse(f); product, pad, border)
end
# function Δ(x, y)
#     sum(abs.(x .- y)) / sum(abs.(y))
# end
