# Main-package side of the FEM machinery.
#
# `stubs.jl` declares the exported function names with helpful error fallbacks
# that fire when the FEM package extension is not loaded.
# `types.jl` defines the FEM-flavoured structs (`FEMDiscretization`,
# `MaternSPDE`, `MaternModel`, the spatiotemporal metadata, etc.) with relaxed
# type parameters so other extensions can reference them whether or not the
# FEM packages are loaded.
#
# Construction logic and methods that actually use Ferrite / Gmsh live in
# `ext/GaussianMarkovRandomFieldsFEM/`.
include("stubs.jl")
include("types.jl")
