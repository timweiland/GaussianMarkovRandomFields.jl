include("backend.jl")
include("gmrf_workspace.jl")
include("cliquetrees_backend.jl")
include("workspace_gmrf.jl")
# gaussian_approximation.jl is included separately in GaussianMarkovRandomFields.jl
# because it depends on observation_models and arithmetic which are loaded later
