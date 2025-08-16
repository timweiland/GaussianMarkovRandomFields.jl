module GaussianMarkovRandomFieldsTests

using GaussianMarkovRandomFields
using ReTest
using Aqua

include("test_utils.jl")  # Shared test utilities
include("test_gmrf.jl")
include("test_metagmrf.jl")
include("linear_maps/test_ad_jacobian.jl")
include("linear_maps/test_zero.jl")
include("linear_maps/test_outer_product.jl")
include("linear_maps/test_linear_map_with_sqrt.jl")
include("linear_maps/test_cholesky_sqrt.jl")
include("linear_maps/test_cholesky_factorized_map.jl")
include("linear_maps/test_symmetric_block_tridiagonal.jl")
include("linear_maps/test_ssm_bidiagonal.jl")
include("preconditioners/test_full_cholesky.jl")
include("preconditioners/test_block_jacobi.jl")
include("preconditioners/test_tridiag_block_gauss_seidel.jl")
include("preconditioners/test_spatiotemporal_preconditioner.jl")
include("spdes/fem/test_fem_discretization.jl")
include("spdes/fem/test_fem_derivatives.jl")
include("mesh/test_scattered.jl")
include("spdes/test_matern.jl")
include("solvers/variance/test_rbmc.jl")
include("test_linearsolve_architecture.jl")
# TODO: Temporarily disabled - Gauss-Newton solvers need LinearSolve.jl integration
# include("optim/test_gauss_newton.jl")
include("test_gmrf_arithmetic.jl")
include("test_mesh_utils.jl")
include("spatiotemporal/test_advection_diffusion.jl")
include("autoregressive/test_car.jl")
# TODO: Temporarily disabled - needs refactoring for LinearSolve.jl integration
# include("ext/test_ldl_factorizations.jl")
include("autodiff/runtests.jl")

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(GaussianMarkovRandomFields; ambiguities = false)
    @test length(Test.detect_ambiguities(GaussianMarkovRandomFields)) == 0
end

end
