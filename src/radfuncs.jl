# using Flux, Functors,Random

"""
radial function
"""

function Radfunc(n=1)
    s= """
Radfunc has been removed to eliminate dependency on Flux. To keep using it, copy below into your program:

using Flux

struct Radfunc
    f
end
@functor Radfunc
Flux.trainable(m::Radfunc) = [m.f]
Flux.trainable(m::Op) = [m.radfunc]

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
"""
# thr
error(s)
end