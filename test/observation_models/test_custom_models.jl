using ForwardDiff
using LinearAlgebra
using SparseArrays
using Distributions
using SparseMatrixColorings, SparseConnectivityTracer
import DifferentiationInterface as DI

@testset "Custom Observation Likelihoods and AD Fallbacks" begin

    @testset "Custom Likelihood with only loglik" begin
        # Define a custom observation likelihood - negative binomial
        struct NegativeBinomialLikelihood <: ObservationLikelihood
            r::Float64  # Number of failures parameter
            y::Vector{Int}  # Observed data
        end

        function GaussianMarkovRandomFields.loglik(x, lik::NegativeBinomialLikelihood)
            μ = exp.(x)  # Mean of negative binomial
            r = lik.r
            y = lik.y
            # Negative binomial parameterization: p = r/(r+μ)
            p = r ./ (r .+ μ)
            return sum(logpdf.(NegativeBinomial.(r, p), y))
        end

        # Provide autodiff backend for automatic gradients/hessians
        GaussianMarkovRandomFields.autodiff_gradient_backend(::NegativeBinomialLikelihood) = DI.AutoForwardDiff()
        GaussianMarkovRandomFields.autodiff_hessian_backend(::NegativeBinomialLikelihood) = DI.AutoSparse(DI.AutoForwardDiff(); sparsity_detector = TracerLocalSparsityDetector(), coloring_algorithm = GreedyColoringAlgorithm())

        # Test the likelihood
        y = [3, 1, 8]
        lik = NegativeBinomialLikelihood(5.0, y)
        x = [1.0, 0.5, 2.0]

        # Test that loglik works
        ll = loglik(x, lik)
        @test ll isa Float64
        @test isfinite(ll)

        # Test that AD fallbacks work for gradient
        grad = loggrad(x, lik)
        grad_fd = ForwardDiff.gradient(xi -> loglik(xi, lik), x)
        @test grad ≈ grad_fd rtol = 1.0e-10

        # Test that AD fallbacks work for hessian
        hess = loghessian(x, lik)
        hess_fd = ForwardDiff.hessian(xi -> loglik(xi, lik), x)
        @test hess ≈ hess_fd rtol = 1.0e-8
    end

    @testset "Custom Likelihood with optimized gradient" begin
        # Custom likelihood that provides its own gradient
        struct CustomNormalLikelihood <: ObservationLikelihood
            σ²::Float64
            y::Vector{Float64}
        end

        function GaussianMarkovRandomFields.loglik(x, lik::CustomNormalLikelihood)
            return -0.5 * sum((lik.y .- x) .^ 2) / lik.σ² - 0.5 * length(lik.y) * log(2π * lik.σ²)
        end

        function GaussianMarkovRandomFields.loggrad(x, lik::CustomNormalLikelihood)
            return (lik.y .- x) ./ lik.σ²
        end

        # Provide gradient backend for automatic hessians
        GaussianMarkovRandomFields.autodiff_gradient_backend(::CustomNormalLikelihood) = DI.AutoForwardDiff()

        y = [0.6, 1.1, -0.4]
        lik = CustomNormalLikelihood(0.25, y)
        x = [0.5, 1.0, -0.5]

        # Test that custom gradient is used
        grad_custom = loggrad(x, lik)
        grad_expected = (y .- x) ./ lik.σ²
        @test grad_custom ≈ grad_expected

        # Test that it matches ForwardDiff
        grad_fd = ForwardDiff.gradient(xi -> loglik(xi, lik), x)
        @test grad_custom ≈ grad_fd rtol = 1.0e-10

        # Test that hessian still uses AD fallback
        hess = loghessian(x, lik)
        hess_fd = ForwardDiff.hessian(xi -> loglik(xi, lik), x)
        @test hess ≈ hess_fd rtol = 1.0e-10
    end

    @testset "Custom Likelihood with both gradient and hessian" begin
        # Simple quadratic likelihood for testing
        struct QuadraticLikelihood <: ObservationLikelihood
            y::Vector{Float64}
        end

        function GaussianMarkovRandomFields.loglik(x, lik::QuadraticLikelihood)
            return -0.5 * sum((x .- lik.y) .^ 2)
        end

        function GaussianMarkovRandomFields.loggrad(x, lik::QuadraticLikelihood)
            return -(x .- lik.y)
        end

        function GaussianMarkovRandomFields.loghessian(x, lik::QuadraticLikelihood)
            return -I(length(x))
        end

        y = [1.1, 1.9, 3.2]
        lik = QuadraticLikelihood(y)
        x = [1.0, 2.0, 3.0]

        # Test custom implementations
        @test loggrad(x, lik) ≈ -(x .- y)
        @test loghessian(x, lik) ≈ -I(3)

        # Verify against ForwardDiff
        grad_fd = ForwardDiff.gradient(xi -> loglik(xi, lik), x)
        @test loggrad(x, lik) ≈ grad_fd

        hess_fd = ForwardDiff.hessian(xi -> loglik(xi, lik), x)
        @test Matrix(loghessian(x, lik)) ≈ hess_fd
    end

    @testset "Sparse Hessian Detection" begin
        # Likelihood that should produce sparse hessian via AutoDiffLikelihood
        function sparse_loglik(x; y)
            # Only neighboring elements interact
            ll = 0.0
            for i in eachindex(y)
                ll += -0.5 * (x[i] - y[i])^2
                if i > 1
                    ll += -0.1 * x[i - 1] * x[i]  # Sparse interaction
                end
            end
            return ll
        end

        # Create AutoDiffObservationModel without hyperparams
        obs_model = AutoDiffObservationModel(sparse_loglik; n_latent = 5)
        y = ones(5) .+ 0.1
        lik = obs_model(y)
        x = ones(5)

        # Test that hessian computation works with automatic sparsity detection
        hess = loghessian(x, lik)
        @test hess isa SparseMatrixCSC

        # Should be close to ForwardDiff result
        hess_fd = ForwardDiff.hessian(xi -> sparse_loglik(xi; y = y), x)
        @test hess ≈ hess_fd rtol = 1.0e-8
    end

    @testset "Error handling" begin
        # Test that missing loglik implementation throws appropriate error
        struct IncompleteLikelihood <: ObservationLikelihood
            y::Vector{Float64}
        end

        y = [1.0]
        lik = IncompleteLikelihood(y)
        x = [1.0]

        @test_throws ErrorException loglik(x, lik)
    end
end
