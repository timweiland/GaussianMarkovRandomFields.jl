module GMRFs

include("typedefs.jl")
include("utils/utils.jl")
include("linear_maps/linear_maps.jl")
include("preconditioners/preconditioners.jl")
include("gmrf.jl")
include("arithmetic/arithmetic.jl")
include("solvers/solvers.jl")
include("autoregressive/autoregressive.jl")
include("spdes/spdes.jl")
include("optim/optim.jl")
include("plots/makie.jl")
include("mesh/mesh.jl")

end
