# using Flux: conv
using LinearAlgebra
using ImageFiltering

function overlapdot(x, f; start=nothing, center=nothing, product=*)
    # start=center===nothing ? ones(Int,ndims(x)) : Int.(center) .-(size(f).-1).÷2
    # @show start,
    if nothing === start === center
        start = ones(Int, ndims(x))
    elseif start === nothing
        start = Int.(center) .- (size(f) .- 1) .÷ 2
    end
    start = Int.(start)

    xl = max.(start, 1)
    xr = min.(start .+ size(f) .- 1, size(x))
    fl = max.(2 .- start, 1)
    fr = min.(size(f), size(x) .- start .+ 1)
    sum(product.(
        x[[a:b for (a, b) in zip(xl, xr)]...],
        f[[a:b for (a, b) in zip(fl, fr)]...],
    ))
end

# function zero_(x)
#     if x <: SVector
#         return zeros(x)
#     end
#     zero(x)
# end

using FFTW
using StaticArrays

"""
    fftconv(x, y; product=*)

Signal processing convolution via FFT in any dimension. 
"""
function fftconv(x, y; product=*)
    @info "convolution using FFT"
    
    sz = size(x) .+ size(y) .- 1
    u = fill(zero(eltype(x)), sz)
    v = fill(zero(eltype(y)), sz)
    @show size(x), size(y),eltype(x),eltype(y)
    u[[1:size(x, i) for i = 1:ndims(x)]...] .= x
    v[[1:size(y, i) for i = 1:ndims(y)]...] .= y
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


function fftconv_(x, y; product=*, pad=:outer, border=0)
    r = fftconv(x, y; product)
    s= ((size(y) .- 1) .÷ 2).+1
    p=size(x)
    I=[a:b for (a,b) in zip(s,s.+p.-1)]
    if pad == :outer && border == 0
        return r
    elseif pad == :same && border == 0
        # return r[[Int(b + 1):Int(a + b) for (a, b) in zip(size(x), (size(y) .- 1) .÷ 2)]...]
        return r[I...]
    elseif pad == :same && border == :circular
        for ix in eachindex(view(r,fill(:,ndims(x))...))
            i=mod.(ix.-s,p).+s
            if i != Tuple(ix)
                r[i...]+=r[ix]
            end
        end
        return r[I...]
    end
    @error "unsupported boundary condition for fft conv"
end
# @show fftconv_([1,0],[1,2,3];pad=:same,border=:circular)
"""
    cvconv(x, f; product = *, stride = 1, pad = 0, alg = nothing)

"convolution" in computer vision for any dimension, same as Cross correlation. Automatically uses FFT for big kernels. For convolution in signal processing , use `dspconv` instead.

`x` input array
`f` filter array
`product` product in convolution, eg `*`, `dot`
`pad` amount of padding or padding option
- any integer number of pixels on each boundary
- `:same`: adds enough padding so ouoverlapdotut is same size as input
- `:outer`: ouoverlapdotut size is `size(x) .+ size(f) .- 1`
`border` type of padding
- `0` value pixels
- `:replicate` repeats edge values
- `:circular` periodic BC
- `:smooth` continuous derivatives at boundaries useful for differential operators
- `:reflect` reflects interior across boundaries which are not repeated
- `:symmetric` same as `:reflect` but with boundaries repeated
`alg` specifies convolution algorithm
- `nothing` Automatically chooses fastest algorithm
- `:direct` convolution, scales as O(n^2)
- `:fft` Fourier convolution, scales as O(n log(n))

Convolutions in other Julia packages, fewer features but perhaps more optimized for speed in their specific use cases
- [ImageFiltering.imfilter](htoverlapdots://juliaimages.org/stable/function_reference/#ImageFiltering.imfilter). Its docs has excellent mathematical explaination of convolutions and correlation as well as padding/border options
- `DSP.conv` `DSP.xcor`
- `Flux.conv`
"""
function cvconv(x, f; product=*, stride=1, pad=0, border=0, alg=nothing)
    if alg===nothing
        if length(f)>27
            alg=:fft
        end
    end

    if alg == :fft
        return fftconv_(x, reverse(f); product, pad, border)
    end
    
    @info "direct convolution "
    @show size(x), size(f),eltype(x),eltype(f)
    if pad == :outer
        pad = size(f) .- 1
    elseif pad == :same
        pad = (size(f) .- 1) .÷ 2
    elseif length(pad) == 1
        pad = fill(pad[1], ndims(x))
    end
    if border == 0
        starts = Iterators.product([
            a:stride:b
            for
            (a, b) in
            zip(ones(Int, ndims(x)) .- pad, size(x) .- size(f) .+ 1 .+ pad)
        ]...)
        # return [overlapdot(x, f;start, product) for l in l]
        return map(start -> overlapdot(x, f; start, product), starts)
    else
        # if border==:circular
        x = parent(padarray(x, Pad(border, pad...)))
        starts = Iterators.product([
            1:stride:b for b in size(x) .- size(f) .+ 1
        ]...)
        return map(
            start -> sum(product.(
                x[[a:b for (a, b) in zip(start, start .+ size(f) .- 1)]...],
                f,
            )), starts
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

Convolution in signal processing. For "convolution" in computer vision, use cvconv instead. Automatically uses FFT for big kernels. By default output size is `size(x) .+ size(f) .- 1`. See `cvconv` for its keyword options which also apply here
"""
function dspconv(x, f; product=*, pad=:outer, border=0, alg=nothing)
    if alg == :fft
        return fftconv_(x, f; product, pad, border)
    end
    cvconv(x, reverse(f); product, pad, border,alg)
end
# function Δ(x, y)
#     sum(abs.(x .- y)) / sum(abs.(y))
# end

function Base.isless(x::Complex, y::Complex)
    abs(x) < abs(y)
end