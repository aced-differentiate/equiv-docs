module EquivariantOperators

include("operators.jl")
include("plotutils.jl")
export Grid, get,put!
export dspconv,cvconv
export Op,Del,Laplacian,Gaussian,Radfunc,remake!,nae
export vector_field_plot
#
end # module
