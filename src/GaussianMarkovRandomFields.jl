module GaussianMarkovRandomFields

include("typedefs.jl")
include("utils/utils.jl")
include("linear_maps/linear_maps.jl")
include("preconditioners/preconditioners.jl")
include("gmrf.jl")
include("metagmrf.jl")
include("observation_models/observation_models.jl")
include("arithmetic/arithmetic.jl")
include("solvers/solvers.jl")
include("autoregressive/autoregressive.jl")
include("spdes/spdes.jl")
include("optim/optim.jl")
include("plots/makie.jl")
include("mesh/mesh.jl")
include("autodiff/autodiff.jl")

end
