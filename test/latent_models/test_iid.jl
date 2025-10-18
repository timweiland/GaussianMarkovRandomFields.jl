using GaussianMarkovRandomFields
using LinearAlgebra
using LinearSolve

@testset "IIDModel" begin
    @testset "Constructor" begin
        model = IIDModel(5)
        @test model.n == 5

        @test_throws ArgumentError IIDModel(0)
        @test_throws ArgumentError IIDModel(-1)
    end

    @testset "Hyperparameters" begin
        model = IIDModel(3)
        params = hyperparameters(model)
        @test params == (τ = Real,)
    end

    @testset "Parameter Validation" begin
        model = IIDModel(3)

        @test precision_matrix(model; τ = 1.0) isa Diagonal
        @test_throws ArgumentError precision_matrix(model; τ = 0.0)
        @test_throws ArgumentError precision_matrix(model; τ = -1.0)
    end

    @testset "Precision Matrix Structure" begin
        model = IIDModel(4)
        τ = 2.5
        Q = precision_matrix(model; τ = τ)

        @test Q isa Diagonal
        @test size(Q) == (4, 4)
        @test all(diag(Q) .== τ)
    end

    @testset "Mean and Constraints" begin
        model = IIDModel(5)
        @test mean(model; τ = 1.0) == zeros(5)
        @test constraints(model; τ = 1.0) === nothing
    end

    @testset "GMRF Construction" begin
        model = IIDModel(3)
        τ = 1.8
        gmrf = model(τ = τ)

        @test gmrf isa GMRF  # Not ConstrainedGMRF
        @test length(gmrf) == 3
        @test precision_matrix(gmrf) == τ * I(3)
    end

    @testset "Type Stability" begin
        model = IIDModel(3)
        Q = precision_matrix(model; τ = 1.0)
        @test eltype(Q) == Float64

        gmrf = model(τ = 1.0)
        @test gmrf isa GMRF{Float64}
    end

    @testset "Algorithm Storage and Passing" begin
        # Test default algorithm (DiagonalFactorization for Diagonal)
        model = IIDModel(10)
        @test model.alg isa DiagonalFactorization

        # Test algorithm is passed to GMRF
        gmrf = model(τ = 1.0)
        @test gmrf.linsolve_cache.alg isa DiagonalFactorization

        # Test custom algorithm
        custom_model = IIDModel(10, alg = CHOLMODFactorization())
        @test custom_model.alg isa CHOLMODFactorization
        custom_gmrf = custom_model(τ = 1.0)
        @test custom_gmrf.linsolve_cache.alg isa CHOLMODFactorization
    end
end
