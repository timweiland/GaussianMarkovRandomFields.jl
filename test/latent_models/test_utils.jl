using GaussianMarkovRandomFields
using Test

@testset "Constraint Processing Utilities" begin
    @testset "_process_constraint with nothing" begin
        result = GaussianMarkovRandomFields._process_constraint(nothing, 10)
        @test result === nothing
    end

    @testset "_process_constraint with :sumtozero" begin
        A, e = GaussianMarkovRandomFields._process_constraint(:sumtozero, 10)
        @test A ≈ ones(1, 10)
        @test e ≈ [0.0]
    end

    @testset "_process_constraint with custom constraint" begin
        A_custom = [1.0 0.0 1.0; 0.0 1.0 1.0]
        e_custom = [1.0, 2.0]
        A, e = GaussianMarkovRandomFields._process_constraint((A_custom, e_custom), 3)
        @test A ≈ A_custom
        @test e ≈ e_custom
    end

    @testset "_process_constraint validation" begin
        # Wrong number of columns
        @test_throws ArgumentError GaussianMarkovRandomFields._process_constraint((ones(1, 5), [0.0]), 10)

        # Mismatched rows and constraint vector length
        @test_throws ArgumentError GaussianMarkovRandomFields._process_constraint((ones(2, 10), [0.0]), 10)

        # Unknown symbol
        @test_throws ArgumentError GaussianMarkovRandomFields._process_constraint(:unknown, 10)
    end
end
