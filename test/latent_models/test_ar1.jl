using GaussianMarkovRandomFields
using LinearAlgebra

@testset "AR1Model" begin
    @testset "Constructor" begin
        # Valid construction
        model = AR1Model(5)
        @test model.n == 5

        # Invalid construction
        @test_throws ArgumentError AR1Model(0)
        @test_throws ArgumentError AR1Model(-1)
    end

    @testset "Hyperparameters" begin
        model = AR1Model(3)
        params = hyperparameters(model)
        @test params == (τ = Real, ρ = Real)
    end

    @testset "Parameter Validation" begin
        model = AR1Model(3)

        # Valid parameters should not throw
        @test precision_matrix(model; τ = 1.0, ρ = 0.5) isa SymTridiagonal
        @test precision_matrix(model; τ = 2.0, ρ = -0.9) isa SymTridiagonal
        @test precision_matrix(model; τ = 0.1, ρ = 0.0) isa SymTridiagonal

        # Invalid τ (must be positive)
        @test_throws ArgumentError precision_matrix(model; τ = 0.0, ρ = 0.5)
        @test_throws ArgumentError precision_matrix(model; τ = -1.0, ρ = 0.5)

        # Invalid ρ (must satisfy |ρ| < 1)
        @test_throws ArgumentError precision_matrix(model; τ = 1.0, ρ = 1.0)
        @test_throws ArgumentError precision_matrix(model; τ = 1.0, ρ = -1.0)
        @test_throws ArgumentError precision_matrix(model; τ = 1.0, ρ = 1.1)
        @test_throws ArgumentError precision_matrix(model; τ = 1.0, ρ = -1.1)
    end

    @testset "Precision Matrix Structure" begin
        @testset "n=1 case" begin
            model = AR1Model(1)
            Q = precision_matrix(model; τ = 2.0, ρ = 0.5)
            @test Q isa SymTridiagonal
            @test size(Q) == (1, 1)
            @test Q[1, 1] == 2.0
        end

        @testset "n=2 case" begin
            model = AR1Model(2)
            τ, ρ = 2.0, 0.5
            Q = precision_matrix(model; τ = τ, ρ = ρ)
            @test Q isa SymTridiagonal
            @test size(Q) == (2, 2)
            @test Q[1, 1] == τ
            @test Q[2, 2] == τ
            @test Q[1, 2] == -ρ * τ
            @test Q[2, 1] == -ρ * τ
        end

        @testset "n=3 case" begin
            model = AR1Model(3)
            τ, ρ = 2.0, 0.5
            Q = precision_matrix(model; τ = τ, ρ = ρ)
            @test Q isa SymTridiagonal
            @test size(Q) == (3, 3)

            # Expected pattern from the plan:
            # [τ      -ρτ      0  ]
            # [-ρτ  (1+ρ²)τ  -ρτ ]
            # [0      -ρτ      τ  ]
            expected = [
                τ           -ρ * τ         0;
                -ρ * τ   (1 + ρ^2) * τ    -ρ * τ;
                0          -ρ * τ         τ
            ]

            @test Matrix(Q) ≈ expected
        end

        @testset "n=5 general case" begin
            model = AR1Model(5)
            τ, ρ = 1.5, 0.7
            Q = precision_matrix(model; τ = τ, ρ = ρ)
            @test Q isa SymTridiagonal
            @test size(Q) == (5, 5)

            # Check main diagonal
            @test Q[1, 1] == τ  # First element
            @test Q[5, 5] == τ  # Last element
            for i in 2:4
                @test Q[i, i] == (1 + ρ^2) * τ  # Middle elements
            end

            # Check off-diagonal
            for i in 1:4
                @test Q[i, i + 1] == -ρ * τ
                @test Q[i + 1, i] == -ρ * τ  # Symmetry
            end

            # Check zeros beyond off-diagonal
            @test Q[1, 3] == 0.0
            @test Q[3, 1] == 0.0
        end
    end

    @testset "Mean Vector" begin
        for n in [1, 3, 10]
            model = AR1Model(n)
            μ = mean(model; τ = 1.0, ρ = 0.5)  # Parameters shouldn't matter for mean
            @test μ == zeros(n)
            @test length(μ) == n
        end
    end

    @testset "Constraints" begin
        model = AR1Model(5)
        constraint_info = constraints(model; τ = 1.0, ρ = 0.5)
        @test constraint_info === nothing
    end

    @testset "GMRF Construction" begin
        model = AR1Model(4)
        τ, ρ = 1.2, 0.6
        gmrf = model(τ = τ, ρ = ρ)

        @test gmrf isa GMRF
        @test length(gmrf) == 4
        @test mean(gmrf) == zeros(4)

        # Check that precision matrix matches our AR1 structure
        Q_expected = precision_matrix(model; τ = τ, ρ = ρ)
        @test precision_matrix(gmrf) == Matrix(Q_expected)
    end

    @testset "Type Stability" begin
        model = AR1Model(3)

        # Test with Float64
        Q_f64 = precision_matrix(model; τ = 1.0, ρ = 0.5)
        @test eltype(Q_f64) == Float64

        # Test with mixed types (should promote)
        Q_mixed = precision_matrix(model; τ = 1, ρ = 0.5)  # Int and Float64
        @test eltype(Q_mixed) == Float64

        # Test type stability of construction
        gmrf = model(τ = 1.0, ρ = 0.5)
        @test gmrf isa GMRF{Float64}
    end
end
