# using Flux: conv
using LinearAlgebra
using ImageFiltering
using FFTW
using StaticArrays
using Memoize
using DSP
# using LRUCache

# @memoize LRU{Tuple{AbstractArray},AbstractArray}(maxsize=25) function fft_(a)
#     println("Running")
#     fft(a)
# end

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

@memoize function padfft_(a, sz)
    @debug "cache miss, computing fft"
    padfft(a, sz)
end

# const FFTCACHE=memoize_cache(padfft_)
# FFTCACHE=memoize_cache(padfft_)
# EQUIVARIANTOPERATORS=(;

setcaching(b) = (global caching = b)
# settings=(;caching=false)
cache = memoize_cache(padfft_)
emptycache() = empty!(cache)

# ENV["EQUIVARIANTOPERATORS_CACHING"]=false

function padfft(a, sz)
    n = length(a[1])
    u = fill(zero(eltype(a)), sz)
    u[[1:size(a, i) for i = 1:ndims(a)]...] .= a
    a = u

    X = fft.([getindex.(a, i) for i in 1:n])
    X = n == 1 ? X[1] : SVector.(X...)
end
"""
fftconv(x, y; product=*)

Signal processing convolution via FFT in any dimension. 
"""
function fftconv(x, y; product=*, caching=false)
    @debug "convolution using FFT"
    # @show size(x), size(y),eltype(x),eltype(y)

    sz = Tuple(size(x) .+ size(y) .- 1)
    f = caching ? padfft_ : padfft
    # @show 1
    X = f(x, sz)
    # @show 2
    Y = f(y, sz)

    Z = product.(X, Y)


    Z = real.(ifft.([getindex.(Z, i) for i in eachindex(Z[1])]))
    nz = length(Z)
    nz == 1 ? Z[1] : SVector.(Z...)
end


function convproc(r,szx,szy; pad=:outer, border=0, caching=false, kw...)
    s = ((szy .- 1) .÷ 2) .+ 1
    p = szx
    I = [a:b for (a, b) in zip(s, s .+ p .- 1)]
    if pad == :outer && border == 0
        return r
    elseif pad == :same && border == 0
        # return r[[Int(b + 1):Int(a + b) for (a, b) in zip(size(x), (size(y) .- 1) .÷ 2)]...]
        return r[I...]
    elseif pad == :same && border in [:circular, :periodic]
        for ix in eachindex(view(r, fill(:, ndims(x))...))
            i = mod.(ix .- s, p) .+ s
            if i != Tuple(ix)
                r[i...] += r[ix]
            end
        end
        return r[I...]
    end end
function fftconv_(x, y; product=*, pad=:outer, border=0, caching=false, kw...)
    r = fftconv(x, y; product, caching)
    
    @error "unsupported boundary condition for fft conv"
end
# @show fftconv_([1,0],[1,2,3];pad=:same,border=:circular)
"""
    cvconv(x, f; product = *, stride = 1, pad = 0, alg = nothing)

"convolution" in computer vision for any dimension, same as Cross correlation. Automatically uses FFT for big kernels. For convolution in signal processing , use `dspconv` instead.

# Args
- `x` input array
- `f` filter array

# Keywords
- `product` product in convolution, eg `*`, `dot`
- `pad` amount of padding or padding option
    - any integer number of pixels on each boundary
    - `:same`: adds enough padding so ouoverlapdotut is same size as input
    - `:outer`: ouoverlapdotut size is `size(x) .+ size(f) .- 1`
- `border` type of padding
    - `0` value pixels
    - `:replicate` repeats edge values
    - `periodic` or `:circular`: periodic BC
    - `:smooth` continuous derivatives at boundaries useful for differential operators
    - `:reflect` reflects interior across boundaries which are not repeated
    - `:symmetric` same as `:reflect` but with boundaries repeated
- `alg` specifies convolution algorithm
    - `nothing` Automatically chooses fastest algorithm
    - `:direct` convolution, scales as O(n^2)
    - `:fft` Fourier convolution, scales as O(n log(n))

Convolutions in other Julia packages, fewer features but perhaps more optimized for speed in their specific use cases
- [ImageFiltering.imfilter](htoverlapdots://juliaimages.org/stable/function_reference/#ImageFiltering.imfilter). Its docs has excellent mathematical explaination of convolutions and correlation as well as padding/border options
- `DSP.conv` `DSP.xcor`
- `Flux.conv`
"""
function cvconv(x::AbstractArray{T}, f::AbstractArray{S}; product=*, stride=1, pad=0, border=0, alg=nothing, periodic=false, kw...) where {T<:Any, S<:Any}
    periodic && (border = :circular)
    if stride == 1
        if alg === nothing
            if length(f) > 27
                alg = :fft
            end
        end
        if (T <: AbstractFloat && S <: AbstractFloat) || alg == :fft
            return dspconv(x, reverse(f); product, pad, border, alg, periodic, kw...)
        end
    end

    @debug "direct convolution "
    # @show size(x), size(f),eltype(x),eltype(f)
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

# function cvconv(x::AbstractArray{T}, f::AbstractArray{T}; kw...) where {T<:Complex}
#     cvconv(x, f; product=(x, y) -> x * conj(y), kw...)
# end
"""
    dspconv(a, f; product = *, pad = :outer, border=0)

Convolution in signal processing. For "convolution" in computer vision, use cvconv instead. Automatically uses FFT for big kernels. By default output size is `size(x) .+ size(f) .- 1`. See `cvconv` for its keyword options which also apply here
"""
function dspconv(x, f; product=*, pad=:outer, border=0, periodic=false, alg=nothing, kw...)
    periodic && (border = :circular)
    if alg === nothing
        if length(f) > 27
            alg = :fft
        end
    end
    if alg == :fft
        return convproc(fftconv(x, f; product),size(x),size(f); pad, border, kw...)
    end
    cvconv(x, reverse(f); product, pad, border, alg, kw...)
end
function dspconv(a::AbstractArray{T}, f::AbstractArray{T}; product=*, kw...) where {T<:AbstractFloat}
    # if product==*
    r = DSP.conv(a, f)
    convproc(r, size(a), size(f); kw...)
end

# function Δ(x, y)
#     sum(abs.(x .- y)) / sum(abs.(y))
# end

function Base.isless(x::Complex, y::Complex)
    abs(x) < abs(y)
end