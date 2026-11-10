using GaussianMarkovRandomFields
using JET

@testset "JET" begin
    if VERSION >= v"1.10" && VERSION < v"1.13-"
        @testset "typo analysis" begin
            JET.test_package(
                GaussianMarkovRandomFields;
                target_modules = (GaussianMarkovRandomFields,),
                mode = :typo,
                toplevel_logger = nothing,
            )
        end
        @testset "error analysis" begin
            JET.test_package(
                GaussianMarkovRandomFields;
                target_modules = (GaussianMarkovRandomFields,),
                toplevel_logger = nothing,
            )
        end
    end
end
