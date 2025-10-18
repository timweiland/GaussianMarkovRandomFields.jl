using GaussianMarkovRandomFields
using LinearAlgebra
using LinearSolve


# Model with constraints
struct ConstrainedTestModel{Alg} <: LatentModel
    n::Int
    alg::Alg
end
ConstrainedTestModel(n::Int; alg = nothing) = ConstrainedTestModel{typeof(alg)}(n, alg)

@testset "LatentModel Interface" begin
    # Test that the abstract interface methods throw appropriate errors
    struct DummyLatentModel <: LatentModel end
    dummy = DummyLatentModel()

    @test_throws ErrorException hyperparameters(dummy)
    @test_throws ErrorException precision_matrix(dummy; τ = 1.0)
    @test_throws ErrorException mean(dummy; τ = 1.0)
    @test_throws ErrorException constraints(dummy; τ = 1.0)
    @test_throws ErrorException model_name(dummy)
    @test_throws ErrorException Base.length(dummy)
end

@testset "LatentModel Call Interface" begin
    # Create a concrete test model that implements the interface
    struct TestLatentModel{Alg} <: LatentModel
        n::Int
        alg::Alg
    end
    TestLatentModel(n::Int; alg = nothing) = TestLatentModel{typeof(alg)}(n, alg)

    GaussianMarkovRandomFields.hyperparameters(::TestLatentModel) = (τ = Real,)

    function GaussianMarkovRandomFields.precision_matrix(model::TestLatentModel; τ::Real)
        return τ * I(model.n)  # Simple identity matrix scaled by τ
    end

    function GaussianMarkovRandomFields.mean(model::TestLatentModel; kwargs...)
        return zeros(model.n)
    end

    function GaussianMarkovRandomFields.constraints(model::TestLatentModel; kwargs...)
        return nothing  # No constraints
    end

    @testset "GMRF Construction" begin
        model = TestLatentModel(3)

        # Test that calling the model creates a GMRF
        gmrf = model(τ = 2.0)

        @test gmrf isa GMRF
        @test length(gmrf) == 3
        @test mean(gmrf) == zeros(3)
        @test precision_matrix(gmrf) == 2.0 * I(3)
    end

    @testset "ConstrainedGMRF Construction" begin

        GaussianMarkovRandomFields.hyperparameters(::ConstrainedTestModel) = (τ = Real,)

        function GaussianMarkovRandomFields.precision_matrix(model::ConstrainedTestModel; τ::Real)
            return τ * I(model.n)
        end

        function GaussianMarkovRandomFields.mean(model::ConstrainedTestModel; kwargs...)
            return zeros(model.n)
        end

        function GaussianMarkovRandomFields.constraints(model::ConstrainedTestModel; kwargs...)
            # Simple constraint: sum of first two elements equals 1
            A = [1.0 1.0 0.0]  # 1×3 matrix
            e = [1.0]
            return (A, e)
        end

        constrained_model = ConstrainedTestModel(3)
        constrained_gmrf = constrained_model(τ = 1.0)

        @test constrained_gmrf isa ConstrainedGMRF
        @test length(constrained_gmrf) == 3
    end
end
