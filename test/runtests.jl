using GMRFs
using Test
using Aqua
using JET

@testset "GMRFs.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(GMRFs; ambiguities = false)
        @test length(Test.detect_ambiguities(GMRFs)) == 0
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(GMRFs; target_defined_modules = true)
    end
    # Write your tests here.
end
