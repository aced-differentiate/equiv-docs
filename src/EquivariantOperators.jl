module EquivariantOperators

include("operators.jl")
include("plotutils.jl")
Laplacian=Lap
export Grid, place!
export dspconv,cvconv
export Op,Del,Lap,Laplacian,Gauss,Radfunc,remake!,nae
export vector_field_plot
export center!,norms,cache,rescale
# export emptycache,setcaching
# export EQUIVARIANTOPERATORS
#
end # module
