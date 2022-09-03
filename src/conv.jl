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
using FFTW
using StaticArrays
function fftconv_(x, y; product=*)
    sz = size(x) .+ size(y) .- 1
    u = zeros(eltype(x[1]), sz)
    v = zeros(eltype(y[1]), sz)
    u[[1:size(x, i) for i = ndims(x)]...] .= x
    v[[1:size(y, i) for i = ndims(y)]...] .= y
    x = u
    y = v

    X = fft.([getindex.(x, i) for i in eachindex(x[1])])
    X = length(X[1][1]) == 1 ? X[1] : SVector.(X...)
    Y = fft.([getindex.(y, i) for i in eachindex(y[1])])
    Y = length(Y[1][1]) == 1 ? Y[1] : SVector.(Y...)
    Z = product.(X, Y)


    r = real.(ifft.([getindex.(Z, i) for i in eachindex(Z[1])]))
    if length(r[1][1]) == 1
        return r[1]
    end
    return SVector.(r...)
    # ifft()
end
function fftconv(x, y; product=*, pad=:outer, border=0)
    r = fftconv_(x, y; product)
    if pad == :outer && border == 0
        return r
    elseif pad == :same && border == 0
        return r[[Int(a):Int(b + a - 1) for (a, b) in zip(size(f), size(x))]...]
    end
    @error "unsupported boundary condition for fft conv"
end

"""
    cvconv(x, f; product = *, stride = 1, pad = 0)

"convolution" in computer vision for any dimension, same as Cross correlation. For convolution in signal processing , use `dspconv` instead.

`x` input array
`f` filter array
`product` product in convolution, eg `*`, `dot`
`pad` amount of padding or padding option
- any integer number of pixels on each boundary
- `:same`: adds enough padding so output is same size as input
- `:outer`: output size is `size(x) .+ size(f) .- 1`
`border` type of padding
- `0` value pixels
- `:replicate` repeats edge values
- `:circular` periodic BC
- `:smooth` continuous derivatives at boundaries useful for differential operators
- `:reflect` reflects interior across boundaries which are not repeated
- `:symmetric` same as `:reflect` but with boundaries repeated

Convolutions in other Julia packages, fewer features but perhaps more optimized for speed in their specific use cases
- [ImageFiltering.imfilter](https://juliaimages.org/stable/function_reference/#ImageFiltering.imfilter). Its docs has excellent mathematical explaination of convolutions and correlation as well as padding/border options
- `DSP.conv` `DSP.xcor`
- `Flux.conv`
"""
function cvconv(x, f; product=*, stride=1, pad=0, border=0, alg=nothing)
    if alg == :fft
        return fftconv(x, reverse(f); product, pad, border)

    end
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
        # return [tp(x, f, l, product) for l in l]
        return map(l -> tp(x, f, l, product), l)
    else
        # if border==:circular
        x = parent(padarray(x, Pad(border, pad...)))
        l = Iterators.product([
            1:stride:b for b in size(x) .- size(f) .+ 1
        ]...)
        return map(
            l -> sum(product.(
                x[[a:b for (a, b) in zip(l, l .+ size(f) .- 1)]...],
                f,
            )), l
        )
        # return [
        #     sum(product.(
        #         x[[a:b for (a, b) in zip(l, l .+ size(f) .- 1)]...],
        #         f,
        #     )) for l in l
        # ]

    end
end

# @show fftconv([1,2],[1,1,1])

function cvconv(x::AbstractArray{T}, f::AbstractArray{T}; kw...) where {T<:Complex}
    cvconv(x, f; product=(x, y) -> x * conj(y), kw...)
end
"""
    dspconv(x, f; product = *,pad = :outer,border=0)

Convolution in signal processing. For "convolution" in computer vision, use cvconv instead. By default output size is `size(x) .+ size(f) .- 1`. See `cvconv` for its keyword options which also apply here
"""
function dspconv(x, f; product=*, pad=:outer, border=0, alg=nothing)
    if alg == :fft
        return fftconv(x, f; product, pad, border)

    end
    cvconv(x, reverse(f); product, pad, border)
end
# function Δ(x, y)
#     sum(abs.(x .- y)) / sum(abs.(y))
# end
