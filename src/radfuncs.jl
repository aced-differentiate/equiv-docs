using Flux, Functors,Random

"""
radial function
"""
struct Radfunc
    f
end
@functor Radfunc
Flux.trainable(m::Radfunc) = [m.f]


function Radfunc(n::Int,σ=leakyrelu)
    Random.seed!(1)
    n=2n
    f = Chain(Dense(1, n, σ), Dense(n, 1))
    Radfunc(f)
end

function (m::Radfunc)(r)
    @unpack f = m
     f([r])[1]
end
