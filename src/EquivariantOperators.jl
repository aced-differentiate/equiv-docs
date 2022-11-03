module EquivariantOperators

include("operators.jl")
include("plotutils.jl")
export Grid, get,put!
export dspconv,cvconv
export Op,Del,Laplacian,Gaussian,Radfunc,remake!,nae
export vector_field_plot
export center!,norms,cache,rescale
# export emptycache,setcaching
# export EQUIVARIANTOPERATORS
#
end # module
