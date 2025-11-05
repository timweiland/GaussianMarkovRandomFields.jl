using Test
using GaussianMarkovRandomFields
import DifferentiationInterface as DI
using ForwardDiff

@testset "AutoDiff Likelihood System" begin

    @testset "Backend fallback and warnings" begin
        # Test hessian backend fallback with warning
        loglik_func = x -> -0.5 * sum(x .^ 2)

        # This should trigger the warning path for AutoDiffObservationModel
        obs_model = AutoDiffObservationModel(
            loglik_func;
            n_latent = 2,
            hessian_backend = nothing
        )
        @test obs_model.hess_backend isa DI.AbstractADType

        # This should trigger the warning path for AutoDiffLikelihood
        likelihood = AutoDiffLikelihood(
            loglik_func;
            n_latent = 2,
            hessian_backend = nothing
        )
        @test likelihood.hess_backend isa DI.AbstractADType
    end

    @testset "Observation model interface" begin
        function test_loglik(x; σ = 1.0, y = [1.0, 2.0])
            return -0.5 * sum((x .- y) .^ 2) / σ^2
        end

        # Use Zygote explicitly to avoid Enzyme closure issues
        obs_model = AutoDiffObservationModel(
            test_loglik;
            n_latent = 2,
            hyperparams = (:σ, :y),
            grad_backend = DI.AutoZygote(),
            hessian_backend = DI.AutoZygote()
        )

        # Test interface methods
        @test latent_dimension(obs_model, [1.0]) == 2
        @test hyperparameters(obs_model) == (:σ, :y)

        # Test materialization with hyperparameters - pass y as positional arg
        y_data = [1.1, 1.9]
        likelihood = obs_model(y_data; σ = 0.5)

        x = [1.0, 2.0]
        ll = loglik(x, likelihood)
        grad = loggrad(x, likelihood)
        hess = loghessian(x, likelihood)

        @test ll isa Float64
        @test length(grad) == 2
        @test size(hess) == (2, 2)

        # Verify correctness
        grad_expected = -(x .- y_data) / 0.25
        @test grad ≈ grad_expected
    end

    @testset "AutoDiff interface methods" begin
        likelihood = AutoDiffLikelihood(x -> sum(x .^ 2); n_latent = 2)

        @test GaussianMarkovRandomFields.autodiff_gradient_backend(likelihood) isa DI.AbstractADType
        @test GaussianMarkovRandomFields.autodiff_hessian_backend(likelihood) isa DI.AbstractADType
        @test GaussianMarkovRandomFields.autodiff_gradient_prep(likelihood) !== nothing
        @test GaussianMarkovRandomFields.autodiff_hessian_prep(likelihood) !== nothing
    end

    @testset "Direct construction paths" begin
        # Test AutoDiffObservationModel with default backends
        simple_model = AutoDiffObservationModel(x -> sum(x .^ 2); n_latent = 3)
        @test simple_model.n_latent == 3
        @test simple_model.hyperparams == ()

        # Test direct AutoDiffLikelihood construction
        simple_lik = AutoDiffLikelihood(x -> sum(x .^ 2); n_latent = 3)
        @test simple_lik isa AutoDiffLikelihood

        x = [1.0, 2.0, 3.0]
        @test loglik(x, simple_lik) == sum(x .^ 2)
    end
end
