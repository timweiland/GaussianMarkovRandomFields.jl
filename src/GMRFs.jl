module GMRFs

include("gmrf.jl")
include("spdes/fem/fem_discretization.jl")
include("spdes/fem/fem_derivatives.jl")
include("spdes/spde.jl")
include("spdes/matern.jl")
include("gmrf_arithmetic.jl")
include("mesh_utils.jl")

end
