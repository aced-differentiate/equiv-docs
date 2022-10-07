using Documenter

include("../src/EquivariantOperators.jl")
using .EquivariantOperators

# include("../src/operators.jl")

##
makedocs(
    sitename = "EquivariantOperators.jl",
    pages = ["index.md","autodiff.md","guide.md",],
    # pages = ["index.md", "architecture.md", "publications.md", "tutorials.md"],
)
