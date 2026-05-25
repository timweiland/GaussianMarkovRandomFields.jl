using GaussianMarkovRandomFields
using JET

@testset "JET" begin
    if VERSION >= v"1.10" && VERSION < v"1.13-"
        @testset "typo analysis" begin
            JET.test_package(
                GaussianMarkovRandomFields;
                target_defined_modules = true,
                mode = :typo,
                toplevel_logger = nothing,
            )
        end
        @testset "error analysis" begin
            JET.test_package(
                GaussianMarkovRandomFields;
                target_defined_modules = true,
                toplevel_logger = nothing,
            )
        end
    end
end
