using GaussianMarkovRandomFields
using Distributions: Normal, Poisson
using SparseArrays
using LinearAlgebra
using Random

@testset "linear_predictor_marginals" begin
    Random.seed!(7)
    n_latent = 6

    # Build a small GMRF posterior for reference. Pattern is fully dense so
    # any LTL we wrap around it has its Hessian sparsity subsumed by Q.
    Q_dense = Symmetric(0.05 * randn(n_latent, n_latent))
    Q_dense = Matrix(Q_dense) + (2.0 + n_latent * 0.05) * I
    Q = sparse(Q_dense)
    μ_prior = randn(n_latent)
    prior = GMRF(μ_prior, Q)
    Σ_ref = inv(Matrix(Q))

    ws_prior = WorkspaceGMRF(μ_prior, Q, GMRFWorkspace(Q))

    @testset "ExponentialFamilyLikelihood (direct, indices = nothing)" begin
        m = ExponentialFamily(Normal)
        y = randn(n_latent)
        lik = m(y; σ = 0.7)

        μ, v, inner = linear_predictor_marginals(prior, lik)
        @test μ ≈ mean(prior)
        @test v ≈ var(prior)
        @test inner === lik
    end

    @testset "ExponentialFamilyLikelihood (direct, with indices)" begin
        idx = [1, 3, 5]
        m = ExponentialFamily(Normal, indices = idx)
        y = randn(length(idx))
        lik = m(y; σ = 0.5)

        μ, v, inner = linear_predictor_marginals(prior, lik)
        @test μ ≈ mean(prior)[idx]
        @test v ≈ var(prior)[idx]
        @test length(μ) == length(idx)
        @test inner === lik
    end

    @testset "LinearlyTransformedLikelihood: μ_η = A μ, v_η = diag(A Σ Aᵀ)" begin
        k = 4
        A = sparse(randn(k, n_latent))
        m = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)
        y = randn(k)
        lik = m(y; σ = 0.3)

        μ_η, v_η, inner = linear_predictor_marginals(prior, lik)
        @test μ_η ≈ A * mean(prior)
        @test v_η ≈ diag(A * Σ_ref * A')
        @test inner === lik.base_likelihood
        @test inner isa GaussianMarkovRandomFields.NormalLikelihood
    end

    @testset "LinearlyTransformedLikelihood on WorkspaceGMRF" begin
        k = 4
        A = sparse(randn(k, n_latent))
        m = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)
        y = randn(k)
        lik = m(y; σ = 0.3)

        μ_η, v_η, inner = linear_predictor_marginals(ws_prior, lik)
        @test μ_η ≈ A * mean(ws_prior)
        @test v_η ≈ diag(A * Σ_ref * A')
        @test inner === lik.base_likelihood
    end

    @testset "CompositeLikelihood: concatenation of per-component results" begin
        idx1 = 1:3
        idx2 = 4:n_latent
        m1 = ExponentialFamily(Normal, indices = idx1)
        m2 = ExponentialFamily(Poisson, indices = idx2)
        composite = CompositeObservationModel((m1, m2))
        y_composite = CompositeObservations((randn(length(idx1)), PoissonObservations(rand(0:5, length(idx2)))))
        lik = composite(y_composite; σ = 0.4)

        μ, v, inner = linear_predictor_marginals(prior, lik)
        @test μ ≈ vcat(mean(prior)[idx1], mean(prior)[idx2])
        @test v ≈ vcat(var(prior)[idx1], var(prior)[idx2])
        @test inner === lik
    end

    @testset "Composite of LinearlyTransformedLikelihoods" begin
        k1, k2 = 3, 2
        A1 = sparse(randn(k1, n_latent))
        A2 = sparse(randn(k2, n_latent))
        m1 = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A1)
        m2 = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A2)
        composite = CompositeObservationModel(
            (m1, m2),
            ((σ = :σ_1,), (σ = :σ_2,)),
        )
        y_composite = CompositeObservations((randn(k1), randn(k2)))
        lik = composite(y_composite; σ_1 = 0.2, σ_2 = 0.6)

        μ, v, inner = linear_predictor_marginals(prior, lik)
        @test μ ≈ vcat(A1 * mean(prior), A2 * mean(prior))
        @test v ≈ vcat(diag(A1 * Σ_ref * A1'), diag(A2 * Σ_ref * A2'))
        @test inner === lik
    end

    @testset "Output shapes and eltypes" begin
        k = 4
        A = sparse(randn(k, n_latent))
        m = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)
        y = randn(k)
        lik = m(y; σ = 0.3)

        μ, v, _ = linear_predictor_marginals(prior, lik)
        @test μ isa AbstractVector
        @test v isa AbstractVector
        @test eltype(μ) == Float64
        @test eltype(v) == Float64
        @test length(μ) == k
        @test length(v) == k
        @test all(>=(0), v)
    end
end
