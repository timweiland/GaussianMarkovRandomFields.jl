using Test
using GaussianMarkovRandomFields
import DifferentiationInterface as DI
using ForwardDiff
using LinearAlgebra: Diagonal, diag
using Distributions: logpdf

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

    @testset "Nested AD through loggrad/loghessian (issue #85)" begin
        # The DI prep cache used to be a single Float64 prep; nested-AD
        # callers (e.g. ForwardDiff over loghessian) hit a
        # PreparationMismatchError. The cache is now eltype-keyed.
        # Pin the inner backends to ForwardDiff so the outer ForwardDiff
        # nests cleanly; default-picked Enzyme can't return Dual values.
        loglik_func = x -> -sum(exp.(x) .- 2 .* x)
        obs_lik = AutoDiffLikelihood(
            loglik_func;
            n_latent = 5,
            grad_backend = DI.AutoForwardDiff(),
            hessian_backend = DI.AutoForwardDiff(),
        )
        x0 = ones(5)
        v = [1.0, 0.0, 0.0, 0.0, 0.0]

        # d/dt sum(loghessian(x0 + t*v)) at t=0 equals -exp(x0[1]) for diagonal H.
        val_h = ForwardDiff.derivative(t -> sum(loghessian(x0 .+ t .* v, obs_lik)), 0.0)
        @test val_h ≈ -exp(x0[1])

        # d/dt sum(loggrad(x0 + t*v)) at t=0 equals -exp(x0[1]).
        val_g = ForwardDiff.derivative(t -> sum(loggrad(x0 .+ t .* v, obs_lik)), 0.0)
        @test val_g ≈ -exp(x0[1])

        # Repeat use exercises the cached Dual prep.
        @test ForwardDiff.derivative(t -> sum(loghessian(x0 .+ t .* v, obs_lik)), 0.0) ≈ val_h

        # Float64 path still produces correct results after the Dual prep is cached.
        @test loggrad(x0, obs_lik) ≈ -(exp.(x0) .- 2)
    end

    @testset "Pointwise-diagonal Hessian fast path (issue #89)" begin
        # When `pointwise_loglik_func` is supplied, `loghessian` should use a
        # per-element second-derivative path that nests cleanly under outer
        # ForwardDiff regardless of the user-selected backends — unblocks
        # nested AD even with default backends that can't return Duals
        # (Enzyme, Mooncake reverse-of-reverse, etc.).
        loglik_func = x -> sum(-(x .- 1.0) .^ 2 .* exp.(x))
        pointwise_func = x -> -(x .- 1.0) .^ 2 .* exp.(x)

        obs_lik = AutoDiffLikelihood(
            loglik_func;
            n_latent = 5,
            pointwise_loglik_func = pointwise_func,
        )
        x0 = ones(5)

        # Float64 baseline matches a direct ForwardDiff.hessian.
        H = loghessian(x0, obs_lik)
        @test H isa Diagonal
        H_ref = ForwardDiff.hessian(loglik_func, x0)
        @test diag(H) ≈ diag(H_ref) rtol = 1.0e-10

        # Nested AD: derivative of trace(H(αx0)) wrt α should match finite diff.
        f = α -> sum(diag(loghessian(α .* x0, obs_lik)))
        g_ad = ForwardDiff.derivative(f, 1.0)
        g_fd = (f(1.0 + 1.0e-5) - f(1.0 - 1.0e-5)) / 2.0e-5
        @test g_ad ≈ g_fd rtol = 1.0e-5

        # No pointwise → falls through to DI.hessian on the joint loglik;
        # eltype-keyed prep cache still handles the Float64 baseline.
        obs_lik_nopointwise = AutoDiffLikelihood(
            loglik_func;
            n_latent = 5,
            grad_backend = DI.AutoForwardDiff(),
            hessian_backend = DI.AutoForwardDiff(),
        )
        H2 = loghessian(x0, obs_lik_nopointwise)
        @test Matrix(H2) ≈ H_ref rtol = 1.0e-10

        # Sensitivity flowing through closure-captured Duals: outer AD
        # differentiates a hyperparameter, but `x` itself is plain Float64.
        # The result eltype must come from the function output, not from x,
        # otherwise the diagonal buffer is wrongly typed and assignment fails.
        function laplace_marginal(log_φ)
            φ = exp(log_φ)
            pointwise = x -> -(x .- 1.0) .^ 2 .* φ
            loglik = x -> sum(pointwise(x))
            obs = AutoDiffLikelihood(
                loglik;
                n_latent = 5,
                grad_backend = DI.AutoForwardDiff(),
                hessian_backend = DI.AutoForwardDiff(),
                pointwise_loglik_func = pointwise,
            )
            return sum(diag(loghessian(ones(5), obs)))
        end
        g_closure_ad = ForwardDiff.derivative(laplace_marginal, 0.0)
        g_closure_fd = (laplace_marginal(1.0e-5) - laplace_marginal(-1.0e-5)) / 2.0e-5
        @test g_closure_ad ≈ g_closure_fd rtol = 1.0e-5
    end

    @testset "OutT type-param dispatch (closure-Dual through gaussian_approximation)" begin
        # End-to-end nested-AD pipeline: outer ForwardDiff differentiates a
        # hyperparameter that's captured by closure into the AutoDiffLikelihood.
        # Newton inside `gaussian_approximation` would crash poking Dual
        # values into the workspace's Float64 Q buffer; the OutT type-param
        # dispatch routes through the existing IFT obs-dual helper instead.
        using SparseArrays: sparse
        using LinearAlgebra: SymTridiagonal

        function laplace_logpdf(log_φ)
            φ = exp(log_φ)
            pointwise = x -> -(x .- 1.0) .^ 2 .* φ
            loglik = x -> sum(pointwise(x))
            obs = AutoDiffLikelihood(
                loglik;
                n_latent = 5,
                grad_backend = DI.AutoForwardDiff(),
                hessian_backend = DI.AutoForwardDiff(),
                pointwise_loglik_func = pointwise,
            )
            Q = sparse(SymTridiagonal(fill(2.0, 5), fill(-0.3, 4)))
            prior = GMRF(zeros(5), Q)
            posterior = gaussian_approximation(prior, obs)
            return logpdf(posterior, zeros(5))
        end

        g_ad = ForwardDiff.derivative(laplace_logpdf, 0.0)
        g_fd = (laplace_logpdf(1.0e-5) - laplace_logpdf(-1.0e-5)) / 2.0e-5
        @test g_ad ≈ g_fd rtol = 1.0e-5
    end
end
