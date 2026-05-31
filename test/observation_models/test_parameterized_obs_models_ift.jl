using GaussianMarkovRandomFields
using Distributions: logpdf
using SparseArrays
using LinearAlgebra
using Random

using DifferentiationInterface
using FiniteDiff, ForwardDiff

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
end
