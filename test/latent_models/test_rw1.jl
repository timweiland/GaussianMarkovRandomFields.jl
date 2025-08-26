using GaussianMarkovRandomFields
using Test
using LinearAlgebra

@testset "RW1Model" begin
    @testset "Constructor" begin
        # Valid construction
        model = RW1Model(5)
        @test model.n == 5
        @test model.regularization == 1.0e-5

        # Custom regularization
        model_custom = RW1Model(3; regularization = 1.0e-4)
        @test model_custom.regularization == 1.0e-4

        # Invalid construction
        @test_throws ArgumentError RW1Model(1)  # n must be > 1
        @test_throws ArgumentError RW1Model(0)
        @test_throws ArgumentError RW1Model(5; regularization = 0.0)  # regularization must be > 0
    end

    @testset "Hyperparameters" begin
        model = RW1Model(3)
        params = hyperparameters(model)
        @test params == (τ = Real,)
    end

    @testset "Parameter Validation" begin
        model = RW1Model(3)

        # Valid parameters should not throw
        @test precision_matrix(model; τ = 1.0) isa SymTridiagonal
        @test precision_matrix(model; τ = 2.0) isa SymTridiagonal

        # Invalid τ (must be positive)
        @test_throws ArgumentError precision_matrix(model; τ = 0.0)
        @test_throws ArgumentError precision_matrix(model; τ = -1.0)
    end

    @testset "Precision Matrix Structure" begin
        @testset "n=3 case" begin
            model = RW1Model(3)  # Use default regularization
            τ = 1.5
            Q = precision_matrix(model; τ = τ)
            @test Q isa SymTridiagonal
            @test size(Q) == (3, 3)

            # Expected RW1 structure scaled by τ plus regularization:
            # [1  -1   0 ]
            # [-1  2  -1 ]
            # [0  -1   1 ]
            expected = τ * [
                1  -1   0;
                -1  2  -1;
                0  -1   1
            ] + model.regularization * I(3)  # Add regularization to diagonal

            @test Matrix(Q) ≈ expected
        end

        @testset "Regularization effect" begin
            model = RW1Model(3; regularization = 1.0e-3)
            τ = 1.0
            Q = precision_matrix(model; τ = τ)

            # The diagonal should have τ * base + regularization
            @test Q[1, 1] == τ + 1.0e-3
            @test Q[2, 2] == 2τ + 1.0e-3
            @test Q[3, 3] == τ + 1.0e-3

            # Off-diagonal unchanged
            @test Q[1, 2] == -τ
        end

        @testset "Scaling by τ" begin
            model = RW1Model(4)  # Use default regularization
            τ = 2.0
            Q = precision_matrix(model; τ = τ)

            # Check main diagonal (scaled by τ plus regularization)
            @test Q[1, 1] == τ + model.regularization    # First element
            @test Q[4, 4] == τ + model.regularization    # Last element
            @test Q[2, 2] == 2τ + model.regularization   # Middle elements
            @test Q[3, 3] == 2τ + model.regularization

            # Check off-diagonal (scaled by τ)
            @test Q[1, 2] == -τ
            @test Q[2, 3] == -τ
            @test Q[3, 4] == -τ
        end
    end

    @testset "Mean Vector" begin
        model = RW1Model(5)
        μ = mean(model; τ = 1.0)
        @test μ == zeros(5)
        @test length(μ) == 5
    end

    @testset "Constraints" begin
        model = RW1Model(5)
        constraint_info = constraints(model; τ = 1.0)
        @test constraint_info !== nothing

        A, e = constraint_info
        @test size(A) == (1, 5)  # 1×n matrix
        @test A == ones(1, 5)    # All ones
        @test e == [0.0]         # Zero constraint (sum-to-zero)
    end

    @testset "ConstrainedGMRF Construction" begin
        model = RW1Model(4)
        τ = 1.2
        gmrf = model(τ = τ)

        # Should return ConstrainedGMRF (not GMRF) due to sum-to-zero constraint
        @test gmrf isa ConstrainedGMRF
        @test length(gmrf) == 4
        @test mean(gmrf) == zeros(4)

        # Check the constraint is correctly applied
        @test gmrf.constraint_matrix == ones(1, 4)
        @test gmrf.constraint_vector == [0.0]
    end

    @testset "Type Stability" begin
        model = RW1Model(3)

        # Test with Float64
        Q_f64 = precision_matrix(model; τ = 1.0)
        @test eltype(Q_f64) == Float64

        # Test type stability of construction
        gmrf = model(τ = 1.0)
        @test gmrf isa ConstrainedGMRF{Float64}
    end
end
