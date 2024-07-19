module GMRFTests

using GMRFs
using ReTest
using Aqua
using JET

include("test_gmrf.jl")
include("spdes/test_fem_discretization.jl")
include("spdes/test_matern.jl")

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(GMRFs; ambiguities = false)
    @test length(Test.detect_ambiguities(GMRFs)) == 0
end
@testset "Code linting (JET.jl)" begin
    JET.test_package(GMRFs; target_defined_modules = true)
end

end
