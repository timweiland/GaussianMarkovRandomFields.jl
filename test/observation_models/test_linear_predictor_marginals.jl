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
        @test loglik(μ, inner) ≈ loglik(mean(prior), lik)
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
        @test inner.indices === nothing
        @test loglik(μ, inner) ≈ loglik(mean(prior), lik)
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
        @test loglik(μ_η, inner) ≈ loglik(mean(prior), lik)
    end

    @testset "LinearlyTransformedLikelihood with offset: μ_η = A μ + b" begin
        k = 4
        A = sparse(randn(k, n_latent))
        b = randn(k)
        m = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A; offset = b)
        y = randn(k)
        lik = m(y; σ = 0.3)

        μ_η, v_η, inner = linear_predictor_marginals(prior, lik)
        @test μ_η ≈ A * mean(prior) .+ b
        # The additive offset shifts the mean but leaves the variance unchanged.
        @test v_η ≈ diag(A * Σ_ref * A')
        @test inner === lik.base_likelihood
        @test loglik(μ_η, inner) ≈ loglik(mean(prior), lik)

        # Regression guard for #154: the offset must actually reach μ_η.
        lik0 = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)(y; σ = 0.3)
        μ_η0, = linear_predictor_marginals(prior, lik0)
        @test μ_η ≈ μ_η0 .+ b
        @test !isapprox(μ_η, μ_η0)
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
        @test inner isa CompositeLikelihood
        @test length(inner.components) == 2
        @test inner.components[1].indices == 1:length(idx1)
        @test inner.components[2].indices == (length(idx1) + 1):(length(idx1) + length(idx2))
        @test loglik(μ, inner) ≈ loglik(mean(prior), lik)
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
        @test inner isa CompositeLikelihood
        @test inner.components[1].indices == 1:k1
        @test inner.components[2].indices == (k1 + 1):(k1 + k2)
        @test loglik(μ, inner) ≈ loglik(mean(prior), lik)
    end

    @testset "ConstrainedGMRF: direct EF uses constrained mean / variance" begin
        A_c = reshape(ones(n_latent), 1, n_latent)
        e = zeros(1)
        ga_c = ConstrainedGMRF(prior, A_c, e)

        m = ExponentialFamily(Normal)
        y = randn(n_latent)
        lik = m(y; σ = 0.7)

        μ, v, _ = linear_predictor_marginals(ga_c, lik)
        @test μ ≈ mean(ga_c)
        @test v ≈ var(ga_c)
    end

    @testset "ConstrainedGMRF: LTL applies constraint correction to v_η" begin
        A_c = reshape(ones(n_latent), 1, n_latent)
        e = zeros(1)
        ga_c = ConstrainedGMRF(prior, A_c, e)

        # Reference Σ_c = Σ - Σ A_c' (A_c Σ A_c')⁻¹ A_c Σ
        Σ_unc = Σ_ref
        Σ_c_ref = Σ_unc - Σ_unc * A_c' * inv(A_c * Σ_unc * A_c') * A_c * Σ_unc

        k = 4
        A = sparse(randn(k, n_latent))
        m = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)
        y = randn(k)
        lik = m(y; σ = 0.3)

        μ_η, v_η, inner = linear_predictor_marginals(ga_c, lik)
        @test μ_η ≈ A * mean(ga_c)
        @test v_η ≈ diag(A * Σ_c_ref * A')
        @test inner === lik.base_likelihood
        # Sanity: constraint correction strictly reduces v_η vs unconstrained.
        v_η_unc, = linear_predictor_marginals(prior, lik)[[2]]
        @test all(v_η .<= v_η_unc .+ 1.0e-10)
    end

    @testset "WorkspaceGMRF with constraints: LTL matches ConstrainedGMRF" begin
        A_c = reshape(ones(n_latent), 1, n_latent)
        e = zeros(1)
        ws_c_prior = WorkspaceGMRF(
            μ_prior, Q, GMRFWorkspace(Q), A_c, e,
        )
        ga_ref = ConstrainedGMRF(prior, A_c, e)

        k = 4
        A = sparse(randn(k, n_latent))
        m = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)
        y = randn(k)
        lik = m(y; σ = 0.3)

        μ_ws, v_ws, _ = linear_predictor_marginals(ws_c_prior, lik)
        μ_ref, v_ref, _ = linear_predictor_marginals(ga_ref, lik)
        @test μ_ws ≈ μ_ref
        @test v_ws ≈ v_ref
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

    @testset "LTL variance via observation-local blocks (#159)" begin
        Random.seed!(159)
        # A sparse banded prior makes the selected inverse Σ genuinely sparse (the
        # Cholesky fill pattern, not a full matrix), so this exercises the fast
        # observation-local path of _row_diag_AΣAt. It must reproduce the original
        # full-product `A * Σ` form exactly on the SAME Σ — for GMRF, WorkspaceGMRF,
        # and ConstrainedGMRF (which reads the unconstrained Σ).
        nl = 40
        Qb = sparse(SymTridiagonal(fill(2.5, nl), fill(-1.0, nl - 1)))
        μb = randn(nl)
        kk = 10
        Ab = sprandn(kk, nl, 0.2)
        A_c = reshape(ones(nl), 1, nl)
        for ga in (
                GMRF(μb, Qb),
                WorkspaceGMRF(μb, Qb, GMRFWorkspace(Qb)),
                ConstrainedGMRF(GMRF(μb, Qb), A_c, zeros(1)),
            )
            fast = GaussianMarkovRandomFields._row_diag_AΣAt(Ab, ga)          # sparse A → fast path
            slow = GaussianMarkovRandomFields._row_diag_AΣAt(Matrix(Ab), ga)  # dense A → A*Σ fallback
            Σ = Matrix(GaussianMarkovRandomFields._posterior_cov_sparse(ga))
            @test fast ≈ slow rtol = 1.0e-10
            @test fast ≈ diag(Ab * Σ * Ab') rtol = 1.0e-10
        end
    end
end
