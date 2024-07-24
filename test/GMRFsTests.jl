module GMRFTests

using GMRFs
using ReTest
using Aqua

include("test_gmrf.jl")
include("spdes/test_fem_discretization.jl")
include("spdes/test_matern.jl")
include("test_gmrf_arithmetic.jl")

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(GMRFs; ambiguities = false)
    @test length(Test.detect_ambiguities(GMRFs)) == 0
end

end
