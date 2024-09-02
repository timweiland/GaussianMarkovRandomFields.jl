module GMRFs

include("typedefs.jl")
include("linear_maps/symmetric_block_tridiagonal.jl")
include("gmrf.jl")
include("linear_conditional_gmrf.jl")
include("spdes/fem/fem_discretization.jl")
include("spdes/fem/fem_derivatives.jl")
include("spdes/fem/utils.jl")
include("linear_ssm.jl")
include("implicit_euler_ssm.jl")
include("spatiotemporal_gmrf.jl")
include("solvers/solver.jl")
include("solvers/cholesky_solver.jl")
include("solvers/cg_solver.jl")
include("solvers/default_solver.jl")
include("spdes/spde.jl")
include("spdes/matern.jl")
include("spdes/advection_diffusion.jl")
include("gmrf_arithmetic.jl")
include("plot_utils.jl")
include("mesh_utils.jl")

end
