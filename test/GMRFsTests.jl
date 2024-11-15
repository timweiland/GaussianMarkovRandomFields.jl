module GMRFTests

using GMRFs
using ReTest
using Aqua

include("test_gmrf.jl")
include("linear_maps/test_ad_jacobian.jl")
include("linear_maps/test_zero.jl")
include("linear_maps/test_outer_product.jl")
include("linear_maps/test_linear_map_with_sqrt.jl")
include("linear_maps/test_cholesky_sqrt.jl")
include("preconditioners/test_full_cholesky.jl")
include("preconditioners/test_block_jacobi.jl")
include("spdes/fem/test_fem_discretization.jl")
include("spdes/fem/test_fem_derivatives.jl")
include("spdes/test_matern.jl")
include("test_gmrf_arithmetic.jl")
include("test_mesh_utils.jl")
include("spatiotemporal/test_advection_diffusion.jl")

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(GMRFs; ambiguities=false)
    @test length(Test.detect_ambiguities(GMRFs)) == 0
end

end
