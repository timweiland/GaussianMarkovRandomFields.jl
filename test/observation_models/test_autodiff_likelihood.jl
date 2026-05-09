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

    @testset "Pointwise Hessian fast path returns Diagonal" begin
        # With BOTH `pointwise_loglik_func` and `diagonal_hessian_safe = true`,
        # loghessian short-circuits to a per-element 1D second-derivative path
        # that returns `Diagonal` directly — bypassing DI.hessian entirely.
        # Doubles as the nested-AD-friendly route since 1D second derivatives
        # nest cleanly.
        using LinearAlgebra: Diagonal

        function loglik_sum(x; y, σ)
            return -0.5 * sum((y .- x) .^ 2) / σ^2
        end
        function loglik_pointwise(x; y, σ)
            return -0.5 .* ((y .- x) .^ 2) ./ σ^2
        end

        obs_model = AutoDiffObservationModel(
            loglik_sum;
            n_latent = 4,
            hyperparams = (:σ,),
            grad_backend = DI.AutoForwardDiff(),
            hessian_backend = DI.AutoForwardDiff(),
            pointwise_loglik_func = loglik_pointwise,
            diagonal_hessian_safe = true,
        )

        y_data = [1.0, 2.0, 3.0, 4.0]
        x = [1.1, 1.9, 3.2, 3.8]
        obs_lik = obs_model(y_data; σ = 0.5)

        H = loghessian(x, obs_lik)
        @test H isa Diagonal
        # H[i,i] = -1/σ² for this Gaussian likelihood
        @test all(diag(H) .≈ -1 / 0.5^2)

        # Nested ForwardDiff over the pointwise loghessian path — should
        # work cleanly because 1D second derivatives don't trip DI's
        # nested-Dual buffer machinery.
        v = [1.0, 0.0, 0.0, 0.0]
        val = ForwardDiff.derivative(t -> sum(loghessian(x .+ t .* v, obs_lik)), 0.0)
        @test val ≈ 0.0  # H is constant in x for the Gaussian case
    end

    @testset "Diagonal-Hessian shortcut is opt-in (issue #102)" begin
        # Pointwise availability does NOT imply diagonal Hessian. For
        # `y[i] ~ Normal(Aᵢᵀ x, σ)` the per-observation term `i` depends on
        # every `x[j]` with `A[i,j] ≠ 0`, so the Hessian is `A' diag(...) A`,
        # not `Diagonal(...)`. With `diagonal_hessian_safe` defaulting to
        # `false`, `loghessian` must return the full (correct) Hessian.
        using LinearAlgebra: Diagonal

        # Linear predictor mixes both latent components in every observation.
        A = [
            1.0 0.5;
            0.5 1.0;
            0.7 0.3;
        ]

        function lp_loglik(x; y, σ, A)
            r = y .- A * x
            return -0.5 * sum(r .^ 2) / σ^2
        end
        function lp_pointwise(x; y, σ, A)
            r = y .- A * x
            return -0.5 .* (r .^ 2) ./ σ^2
        end

        obs_model_unsafe = AutoDiffObservationModel(
            lp_loglik;
            n_latent = 2,
            hyperparams = (:σ, :A),
            grad_backend = DI.AutoForwardDiff(),
            hessian_backend = DI.AutoForwardDiff(),
            pointwise_loglik_func = lp_pointwise,
            # diagonal_hessian_safe defaults to `false`
        )

        y_data = [1.0, 2.0, 1.5]
        σ = 0.5
        x = [0.7, 1.1]
        obs_lik_unsafe = obs_model_unsafe(y_data; σ = σ, A = A)

        H_full = loghessian(x, obs_lik_unsafe)
        H_expected = -A' * A / σ^2

        # Default path returns the full Hessian, not a diagonal stub.
        @test !(H_full isa Diagonal)
        @test H_full ≈ H_expected
        # Off-diagonal must be present and nonzero — the bug was returning 0 here.
        @test abs(H_full[1, 2]) > 1.0e-3

        # Sanity: diagonal entries also differ from what the broken shortcut
        # would have produced (the shortcut only sums one observation term per i).
        # H_full[1,1] = -sum(A[:,1].^2)/σ²; broken[1,1] would have been -A[1,1]²/σ².
        @test H_full[1, 1] ≈ -sum(A[:, 1] .^ 2) / σ^2
        @test H_full[1, 1] != -A[1, 1]^2 / σ^2

        # Opting in flips back to the (now wrong-for-this-model) diagonal path.
        # Verifies the gate, not correctness — user is asserting the safety.
        obs_model_optedin = AutoDiffObservationModel(
            lp_loglik;
            n_latent = 2,
            hyperparams = (:σ, :A),
            grad_backend = DI.AutoForwardDiff(),
            hessian_backend = DI.AutoForwardDiff(),
            pointwise_loglik_func = lp_pointwise,
            diagonal_hessian_safe = true,
        )
        obs_lik_optedin = obs_model_optedin(y_data; σ = σ, A = A)
        @test loghessian(x, obs_lik_optedin) isa Diagonal
    end
end
