using LinearAlgebra
using ForwardDiff
using Distributions

@testset "LinearlyTransformedObservationModel" begin

    @testset "Basic Construction" begin
        base_model = ExponentialFamily(Poisson)
        A = [1.0 0.5; 1.0 1.0]  # 2 obs, 2 latent components

        ltom = LinearlyTransformedObservationModel(base_model, A)
        @test ltom.base_model === base_model
        @test ltom.design_matrix === A

        # Test hyperparameter delegation
        @test hyperparameters(ltom) == ()  # Poisson has no hyperparameters
    end

    @testset "Materialization" begin
        base_model = ExponentialFamily(Normal)
        A = sparse(1.0 * I, 2, 2)  # Identity
        ltom = LinearlyTransformedObservationModel(base_model, A)

        y = [1.0, 2.0]
        ltlik = ltom(y; σ = 1.0)

        @test ltlik isa LinearlyTransformedLikelihood
        @test ltlik.design_matrix === A
    end

    @testset "Chain Rule Verification" begin
        base_model = ExponentialFamily(Normal)
        A = sparse([1.0 0.5; 0.0 1.0])
        ltom = LinearlyTransformedObservationModel(base_model, A)

        y = [1.0, 2.0]
        ltlik = ltom(y; σ = 1.0)
        x_full = [0.5, 1.0]

        # Verify chain rule with ForwardDiff
        grad_fd = ForwardDiff.gradient(x -> loglik(x, ltlik), x_full)
        grad_analytical = loggrad(x_full, ltlik)
        @test grad_analytical ≈ grad_fd

        hess_fd = ForwardDiff.hessian(x -> loglik(x, ltlik), x_full)
        hess_analytical = loghessian(x_full, ltlik)
        @test hess_analytical ≈ hess_fd
    end

    @testset "Identity Matrix Equivalence" begin
        # When A = I, should match base model exactly
        base_model = ExponentialFamily(Poisson)
        A = sparse(1.0 * I, 2, 2)
        ltom = LinearlyTransformedObservationModel(base_model, A)

        y = PoissonObservations([1, 3])
        ltlik = ltom(y)
        base_lik = base_model(y)

        x = [0.5, 1.0]
        @test loglik(x, ltlik) ≈ loglik(x, base_lik)
    end

    @testset "Conditional Distribution" begin
        base_model = ExponentialFamily(Poisson)
        A = sparse([1.0 0.5; 0.0 1.0])
        ltom = LinearlyTransformedObservationModel(base_model, A)

        x_full = [0.5, 1.0]
        dist = conditional_distribution(ltom, x_full)

        @test dist isa Distribution
        @test length(dist) == 2

        y = rand(dist)
        @test all(y .>= 0)  # Count data
    end

end
