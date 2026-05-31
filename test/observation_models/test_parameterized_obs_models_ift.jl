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
