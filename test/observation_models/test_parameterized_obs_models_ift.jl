using GaussianMarkovRandomFields
using Distributions: logpdf
using SparseArrays
using LinearAlgebra
using Random

using DifferentiationInterface
using FiniteDiff, ForwardDiff
using Zygote
import ChainRulesCore

# ForwardDiff hyperparameter sensitivities through `gaussian_approximation` when
# the LinearlyTransformedObservationModel's design matrix depends on θ. The
# materialized matrix carries Dual partials, and the existing forward-mode IFT
# machinery threads them through both the WorkspaceGMRF and plain GMRF{Dual}
# paths. Cross-checked against finite differences.

@testset "Parameterized design matrix — ForwardDiff IFT" begin
    fd = AutoFiniteDiff()

    @testset "Dual design matrix, WorkspaceGMRF{Float64} prior" begin
        Random.seed!(1)
        n = 5
        model = AR1Model(n)
        y = randn(n)
        z = zeros(n)
        base = ExponentialFamily(Normal)
        build_A(; κ) = sparse(Diagonal(fill(κ, n)))   # diagonal ⇒ pattern ⊆ AR1 prior
        ltom = LinearlyTransformedObservationModel(
            base, ParameterizedMatrix(build_A; hyperparameters = (:κ,), n_latent = n)
        )
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)

        function pipe(θ)
            prior = model(ws; τ = 1.0, ρ = 0.3)         # WorkspaceGMRF{Float64}
            lik = ltom(y; σ = 1.0, κ = θ[1])            # Dual κ ⇒ Dual A
            post = gaussian_approximation(prior, lik)
            return logpdf(post, z)
        end

        θ0 = [0.8]
        g_fwd = DifferentiationInterface.gradient(pipe, AutoForwardDiff(), θ0)
        g_fd = DifferentiationInterface.gradient(pipe, fd, θ0)
        @test maximum(abs.(g_fwd - g_fd)) < 1.0e-3
    end

    @testset "Dual design matrix + Dual prior, GMRF{Dual} plain path" begin
        Random.seed!(2)
        n = 5
        model = AR1Model(n)
        y = randn(n)
        z = zeros(n)
        base = ExponentialFamily(Normal)
        build_A(; κ) = sparse(Diagonal(fill(κ, n)))
        ltom = LinearlyTransformedObservationModel(
            base, ParameterizedMatrix(build_A; hyperparameters = (:κ,), n_latent = n)
        )

        function pipe(θ)
            μ = mean(model; τ = exp(θ[1]), ρ = 0.3)
            Q = sparse(precision_matrix(model; τ = exp(θ[1]), ρ = 0.3))
            prior = GMRF(μ, Q)                          # GMRF{Dual}
            lik = ltom(y; σ = 1.0, κ = θ[2])            # Dual A
            post = gaussian_approximation(prior, lik)
            return logpdf(post, z)
        end

        θ0 = [0.1, 0.9]
        g_fwd = DifferentiationInterface.gradient(pipe, AutoForwardDiff(), θ0)
        g_fd = DifferentiationInterface.gradient(pipe, fd, θ0)
        @test maximum(abs.(g_fwd - g_fd)) < 1.0e-3
    end

    @testset "θ-varying sparsity pattern is rejected" begin
        Random.seed!(3)
        n = 4
        model = AR1Model(n)
        y = randn(n)
        base = ExponentialFamily(Normal)
        # off-diagonal (3,1) entry: A'A pattern exceeds the tridiagonal AR1 prior
        build_A(; κ) = sparse([1.0 0 0 0; 0 1 0 0; κ 0 1 0; 0 0 0 1])
        ltom = LinearlyTransformedObservationModel(
            base, ParameterizedMatrix(build_A; hyperparameters = (:κ,), n_latent = n)
        )
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)   # tridiagonal prior pattern

        prior = model(ws; τ = 1.0, ρ = 0.3)
        lik = ltom(y; σ = 1.0, κ = ForwardDiff.Dual(0.5, 1.0))
        @test_throws Exception gaussian_approximation(prior, lik)
    end
end

# NonlinearLeastSquares residual hyperparameters. The residual Jacobian's sparsity
# pattern is fixed at materialization, so the Gauss–Newton loggrad/loghessian compose
# with an outer ForwardDiff pass. The IFT mode-sensitivity solve uses the true Hessian
# (Gauss–Newton precision corrected by the residual-curvature term Σₖ (W r)ₖ ∇²fₖ), so
# gradients are exact for residuals nonlinear in the latent field too.
@testset "Parameterized NLSQ residual — ForwardDiff IFT" begin
    fd = AutoFiniteDiff()

    @testset "Nonlinear-in-x residual, Dual α (workspace IFT)" begin
        Random.seed!(11)
        n = 5
        model = AR1Model(n)
        y = randn(n)
        z = zeros(n)
        f = (x; α) -> α .* x .+ 0.1 .* x .^ 2        # nonlinear in x ⇒ exercises the correction
        nlsq = NonlinearLeastSquaresModel(f, n; hyperparams = (:α,))
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)

        function pipe(θ)
            prior = model(ws; τ = 1.0, ρ = 0.3)
            lik = nlsq(y; σ = 0.5, α = θ[1])
            return logpdf(gaussian_approximation(prior, lik), z)
        end

        θ0 = [1.2]
        g_fwd = DifferentiationInterface.gradient(pipe, AutoForwardDiff(), θ0)
        g_fd = DifferentiationInterface.gradient(pipe, fd, θ0)
        # Tolerance set by the finite-difference reference's accuracy (~1e-3). Without
        # the residual-curvature correction the Gauss–Newton-only gradient is off by
        # ~1e-1 here, so this firmly tests the correction.
        @test maximum(abs.(g_fwd - g_fd)) < 5.0e-3
    end

    @testset "Dual σ + Dual α (workspace IFT)" begin
        Random.seed!(12)
        n = 5
        model = AR1Model(n)
        y = randn(n)
        z = zeros(n)
        f = (x; α) -> α .* x
        nlsq = NonlinearLeastSquaresModel(f, n; hyperparams = (:α,))
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)

        function pipe(θ)
            prior = model(ws; τ = 1.0, ρ = 0.3)
            lik = nlsq(y; σ = exp(θ[1]), α = θ[2])
            return logpdf(gaussian_approximation(prior, lik), z)
        end

        θ0 = [-0.3, 1.1]
        g_fwd = DifferentiationInterface.gradient(pipe, AutoForwardDiff(), θ0)
        g_fd = DifferentiationInterface.gradient(pipe, fd, θ0)
        @test maximum(abs.(g_fwd - g_fd)) < 1.0e-3
    end

    @testset "Non-diagonal Jacobian, joint-pattern workspace" begin
        Random.seed!(13)
        n = 5
        model = AR1Model(n)
        y = randn(n - 1)
        z = zeros(n)
        f = (x; α) -> [α * x[i] - x[i + 1] for i in 1:(n - 1)]   # bidiagonal J, linear in x
        nlsq = NonlinearLeastSquaresModel(f, n; hyperparams = (:α,))
        ref = nlsq(y; σ = 0.5, α = 1.0)
        ws = GMRFWorkspace(model, ref; τ = 1.0, ρ = 0.3)         # joint prior+Hessian pattern

        function pipe(θ)
            prior = model(ws; τ = 1.0, ρ = 0.3)
            lik = nlsq(y; σ = 0.5, α = θ[1])
            return logpdf(gaussian_approximation(prior, lik), z)
        end

        θ0 = [1.4]
        g_fwd = DifferentiationInterface.gradient(pipe, AutoForwardDiff(), θ0)
        g_fd = DifferentiationInterface.gradient(pipe, fd, θ0)
        @test maximum(abs.(g_fwd - g_fd)) < 1.0e-3
    end

    @testset "Reverse-mode through NLSQ errors clearly" begin
        n = 4
        prior = GMRF(zeros(n), sparse(SymTridiagonal(fill(2.5, n), fill(-1.0, n - 1))))
        nlsq = NonlinearLeastSquaresModel((x; α) -> α .* x, n; hyperparams = (:α,))
        lik = nlsq(randn(n); σ = 0.5, α = 1.0)
        cfg = Zygote.ZygoteRuleConfig()
        # The reverse-mode rrule must reject NLSQ up front (rather than failing deep in
        # AD internals) for both plain GMRF and WorkspaceGMRF priors.
        @test_throws ArgumentError ChainRulesCore.rrule(cfg, gaussian_approximation, prior, lik)
        ws = make_workspace(AR1Model(n); τ = 1.0, ρ = 0.3)
        wprior = AR1Model(n)(ws; τ = 1.0, ρ = 0.3)
        @test_throws ArgumentError ChainRulesCore.rrule(cfg, gaussian_approximation, wprior, lik)
    end

    # The residual's sparse-AD backends (Jacobian + residual-curvature Hessian) have
    # θ-/x-independent patterns, so they are detected once per model and reused across
    # materializations rather than re-detected on every `model(y; …)` call / gradient pass.
    @testset "Sparse-AD backends detected once and cached on the model" begin
        Random.seed!(21)
        n = 6
        # Residual coupling neighbouring latents ⇒ genuinely non-diagonal Hessian pattern.
        f = (x; α) -> α .* x .^ 2 .+ vcat(x[2:n], x[1]) .* x
        model = NonlinearLeastSquaresModel(f, n; hyperparams = (:α,))
        y = randn(n)

        # First materialization populates the cache; a second (different σ and α) reuses it.
        @test model.backend_cache.jacobian === nothing
        @test model.backend_cache.hessian === nothing
        lik1 = model(y; σ = 0.5, α = 1.3)
        @test model.backend_cache.jacobian !== nothing
        @test model.backend_cache.hessian !== nothing
        lik2 = model(y; σ = 0.9, α = 2.1)
        @test lik2.jac_backend === lik1.jac_backend     # identical object ⇒ no re-detection
        @test lik2.hess_backend === lik1.hess_backend

        # The cached residual-Hessian backend yields the correct curvature: compare the
        # sparse result to a dense ForwardDiff Hessian of x -> Σ_k (W r)_k f_k(x). Checking
        # at two structurally different linearization points (the pattern is detected once,
        # at zeros) guards that the cached pattern is a valid superset away from the probe —
        # which holds because this residual's sparsity is x-independent (the documented
        # contract; data-dependent structure-changing branches are unsupported).
        fα = x -> f(x; α = 1.3)
        for x_star in (randn(n), abs.(randn(n)) .+ 1.0)
            Wr = lik1.inv_σ² .* (lik1.y .- fα(x_star))
            C_cached = GaussianMarkovRandomFields.residual_curvature(lik1, x_star)
            C_ref = ForwardDiff.hessian(x -> sum(Wr .* fα(x)), x_star)
            @test Matrix(C_cached) ≈ C_ref
        end

        # End-to-end: the hyperparameter gradient through gaussian_approximation still
        # matches finite differences (caching the pattern doesn't change the numbers).
        prior_model = AR1Model(n)
        ws = make_workspace(prior_model; τ = 1.0, ρ = 0.3)
        z = zeros(n)
        function pipe(θ)
            prior = prior_model(ws; τ = 1.0, ρ = 0.3)
            lik = model(y; σ = 0.5, α = θ[1])
            return logpdf(gaussian_approximation(prior, lik), z)
        end
        θ0 = [1.3]
        g_fwd = DifferentiationInterface.gradient(pipe, AutoForwardDiff(), θ0)
        g_fd = DifferentiationInterface.gradient(pipe, AutoFiniteDiff(), θ0)
        @test maximum(abs.(g_fwd - g_fd)) < 5.0e-3
    end
end

# Affine offset hyperparameter sensitivities. η = A·x + b is affine in x, so the
# offset enters the IFT through the (cheap) offset builder only: it shifts the mode and,
# for non-Gaussian bases, the point at which the base Hessian is evaluated. Exactness is
# cross-checked against finite differences for Gaussian (offset-invariant Hessian) and
# Poisson (offset-shifted Hessian) bases.
@testset "Parameterized offset — ForwardDiff IFT" begin
    fd = AutoFiniteDiff()

    @testset "Normal base, Dual offset (workspace IFT)" begin
        Random.seed!(21)
        n = 5
        model = AR1Model(n)
        A = sparse(1.0 * I, n, n)
        build_b(; s) = fill(s, n)
        ltom = LinearlyTransformedObservationModel(
            ExponentialFamily(Normal), A;
            offset = ParameterizedOffset(build_b; hyperparameters = (:s,))
        )
        y = randn(n)
        z = zeros(n)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        function pipe(θ)
            prior = model(ws; τ = 1.0, ρ = 0.3)
            lik = ltom(y; σ = 1.0, s = θ[1])
            return logpdf(gaussian_approximation(prior, lik), z)
        end
        gF = DifferentiationInterface.gradient(pipe, AutoForwardDiff(), [0.4])
        gD = DifferentiationInterface.gradient(pipe, fd, [0.4])
        @test maximum(abs.(gF - gD)) < 1.0e-3
    end

    @testset "Poisson base, Dual offset shifts the Hessian (workspace IFT)" begin
        Random.seed!(22)
        n = 5
        model = AR1Model(n)
        A = sparse(1.0 * I, n, n)
        build_b(; s) = fill(s, n)
        ltom = LinearlyTransformedObservationModel(
            ExponentialFamily(Poisson), A;
            offset = ParameterizedOffset(build_b; hyperparameters = (:s,))
        )
        y = PoissonObservations([2, 1, 3, 0, 4])
        z = zeros(n)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        function pipe(θ)
            prior = model(ws; τ = 1.0, ρ = 0.3)
            lik = ltom(y; s = θ[1])
            return logpdf(gaussian_approximation(prior, lik), z)
        end
        gF = DifferentiationInterface.gradient(pipe, AutoForwardDiff(), [0.2])
        gD = DifferentiationInterface.gradient(pipe, fd, [0.2])
        @test maximum(abs.(gF - gD)) < 1.0e-3
    end

    @testset "Joint Dual offset + Dual design matrix" begin
        Random.seed!(23)
        n = 5
        model = AR1Model(n)
        build_A(; κ) = sparse(Diagonal(fill(κ, n)))
        build_b(; s) = fill(s, n)
        ltom = LinearlyTransformedObservationModel(
            ExponentialFamily(Normal),
            ParameterizedMatrix(build_A; hyperparameters = (:κ,), n_latent = n);
            offset = ParameterizedOffset(build_b; hyperparameters = (:s,))
        )
        @test hyperparameters(ltom) == (:σ, :κ, :s)
        y = randn(n)
        z = zeros(n)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        function pipe(θ)
            prior = model(ws; τ = 1.0, ρ = 0.3)
            lik = ltom(y; σ = 1.0, κ = θ[1], s = θ[2])
            return logpdf(gaussian_approximation(prior, lik), z)
        end
        gF = DifferentiationInterface.gradient(pipe, AutoForwardDiff(), [0.8, 0.4])
        gD = DifferentiationInterface.gradient(pipe, fd, [0.8, 0.4])
        @test maximum(abs.(gF - gD)) < 1.0e-3
    end

    @testset "Dual offset + Dual prior (GMRF{Dual} plain path)" begin
        Random.seed!(24)
        n = 5
        model = AR1Model(n)
        A = sparse(1.0 * I, n, n)
        build_b(; s) = fill(s, n)
        ltom = LinearlyTransformedObservationModel(
            ExponentialFamily(Normal), A;
            offset = ParameterizedOffset(build_b; hyperparameters = (:s,))
        )
        y = randn(n)
        z = zeros(n)
        function pipe(θ)
            μ = mean(model; τ = exp(θ[1]), ρ = 0.3)
            Q = sparse(precision_matrix(model; τ = exp(θ[1]), ρ = 0.3))
            prior = GMRF(μ, Q)
            lik = ltom(y; σ = 1.0, s = θ[2])
            return logpdf(gaussian_approximation(prior, lik), z)
        end
        gF = DifferentiationInterface.gradient(pipe, AutoForwardDiff(), [0.1, 0.4])
        gD = DifferentiationInterface.gradient(pipe, fd, [0.1, 0.4])
        @test maximum(abs.(gF - gD)) < 1.0e-3
    end
end
