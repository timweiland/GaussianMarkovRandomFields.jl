using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Random
using Distributions: Poisson, mean, var
using LinearSolve
using ReTest: @testset, @test

# Regression coverage for conditioning latent models whose precision matrix is
# `SymTridiagonal` (AR1Model / RW1Model), which get an `LDLtFactorization`.
#
# Conditioning turns that precision into a sparse matrix, and `LDLtFactorization` bottoms
# out in `LinearAlgebra.ldlt!`, whose only bare-matrix method is `ldlt!(::SymTridiagonal)`.
# Inheriting the prior's algorithm therefore threw for the most basic workflow the package
# has. Both conditioning entry points are covered here, since they failed differently:
# `linear_condition` through the `GMRF` constructor, and `gaussian_approximation` through
# in-place reuse of the prior's LinearSolve cache.
@testset "SymTridiagonal prior conditioning" begin
    # Dense reference posterior for x ~ N(μ, Q⁻¹) observed as y = A x + ϵ, ϵ ~ N(0, Q_ϵ⁻¹)
    function dense_posterior(μ, Q, A, Q_ϵ, y)
        Q_post = Matrix(Q) + A' * Q_ϵ * A
        Σ = inv(Q_post)
        return Σ * (Matrix(Q) * μ + A' * Q_ϵ * y), Σ
    end

    @testset "AR1Model linear_condition" begin
        n = 40
        prior = AR1Model(n)(τ = 1.5, ρ = 0.8)
        @test precision_map(prior) isa SymTridiagonal
        @test prior.linsolve_cache.alg isa LinearSolve.LDLtFactorization

        # Observe three interior points with different noise precisions
        idx = [7, 19, 33]
        A = sparse(1:3, idx, ones(3), 3, n)
        Q_ϵ = Diagonal([100.0, 25.0, 400.0])
        y = [0.5, -0.25, 1.0]

        post = linear_condition(prior; A = A, Q_ϵ = Q_ϵ, y = y)

        μ_ref, Σ_ref = dense_posterior(mean(prior), precision_matrix(prior), Matrix(A), Matrix(Q_ϵ), y)
        @test mean(post) ≈ μ_ref
        @test var(post) ≈ diag(Σ_ref)
        @test precision_matrix(post) ≈ Matrix(precision_matrix(prior)) + Matrix(A)' * Q_ϵ * Matrix(A)

        # Sampling must work on the posterior, both single- and multi-draw
        rng = MersenneTwister(42)
        @test length(rand(rng, post)) == n
        @test size(rand(rng, post, 4)) == (n, 4)
    end

    @testset "RW1Model linear_condition" begin
        n = 25
        prior = RW1Model(n)(τ = 2.0)
        # RW1Model carries a sum-to-zero constraint, so the prior is a ConstrainedGMRF;
        # the LDLtFactorization lives on the base GMRF.
        @test prior isa ConstrainedGMRF
        @test precision_map(prior.base_gmrf) isa SymTridiagonal
        @test prior.base_gmrf.linsolve_cache.alg isa LinearSolve.LDLtFactorization

        idx = [4, 12, 21]
        A = sparse(1:3, idx, ones(3), 3, n)
        Q_ϵ = Diagonal(fill(50.0, 3))
        y = [1.0, 0.0, -1.0]

        post = linear_condition(prior; A = A, Q_ϵ = Q_ϵ, y = y)
        @test post isa ConstrainedGMRF

        # Dense reference: condition the *base* (unconstrained) GMRF, then apply the
        # sum-to-zero constraint via the Kriging correction the ConstrainedGMRF documents.
        μ_u, Σ_u = dense_posterior(
            mean(prior.base_gmrf), precision_matrix(prior.base_gmrf),
            Matrix(A), Matrix(Q_ϵ), y
        )
        C = ones(1, n)                      # sum-to-zero constraint matrix
        e = [0.0]
        S = C * Σ_u * C'
        μ_c = μ_u - Σ_u * C' * (S \ (C * μ_u - e))
        Σ_c = Σ_u - Σ_u * C' * (S \ (C * Σ_u))

        @test mean(post) ≈ μ_c
        @test var(post) ≈ diag(Σ_c)
        @test abs(sum(mean(post))) < 1.0e-10
    end

    @testset "Fast path: SymTridiagonal posterior keeps LDLtFactorization" begin
        # A diagonal observation contribution leaves the precision tridiagonal, so the
        # specialized algorithm must be retained rather than falling back.
        n = 30
        prior = AR1Model(n)(τ = 1.0, ρ = 0.7)
        post = linear_condition(
            prior; A = 1.0 * I, Q_ϵ = 3.0, y = zeros(n), b = zeros(n),
            obs_precision_contrib = Diagonal(fill(3.0, n))
        )
        @test precision_map(post) isa SymTridiagonal
        @test post.linsolve_cache.alg isa LinearSolve.LDLtFactorization

        Σ_ref = inv(Matrix(precision_matrix(prior)) + 3.0I)
        @test var(post) ≈ diag(Σ_ref)
    end

    @testset "gaussian_approximation with a sparse Hessian" begin
        # A non-identity design matrix makes the observation Hessian sparse rather than
        # Diagonal, so the posterior precision leaves the SymTridiagonal storage type and
        # the prior's cache can no longer be reused in place.
        n = 12
        prior = AR1Model(n)(τ = 1.0, ρ = 0.85)
        idx = [1, 3, 5, 7, 9]
        A = sparse(1:5, idx, ones(5), 5, n)
        lik = LinearlyTransformedLikelihood(
            ExponentialFamily(Poisson)(PoissonObservations([2, 1, 3, 2, 1])), A
        )

        post = gaussian_approximation(prior, lik)
        @test post isa GMRF

        # The returned mean must be the posterior mode: ∇ log p(x|y) = 0 there.
        grad = GaussianMarkovRandomFields.gradlogpdf(prior, mean(post)) .+
            GaussianMarkovRandomFields.loggrad(mean(post), lik)
        @test norm(grad) < 1.0e-8

        # ...and the precision must be the negative Hessian of the neg-log-posterior.
        Q_expected = Matrix(precision_matrix(prior)) -
            Matrix(GaussianMarkovRandomFields.loghessian(mean(post), lik))
        @test Matrix(precision_matrix(post)) ≈ Q_expected
    end

    @testset "gaussian_approximation fast path: Diagonal Hessian" begin
        # An identity design keeps the Hessian Diagonal, so `Q - H` stays SymTridiagonal
        # and the prior's LDLt cache should still be reused.
        n = 10
        prior = AR1Model(n)(τ = 1.0, ρ = 0.85)
        lik = ExponentialFamily(Poisson)(PoissonObservations([2, 1, 3, 2, 1, 4, 2, 1, 2, 3]))

        post = gaussian_approximation(prior, lik)
        @test precision_matrix(post) isa SymTridiagonal
        @test post.linsolve_cache.alg isa LinearSolve.LDLtFactorization

        grad = GaussianMarkovRandomFields.gradlogpdf(prior, mean(post)) .+
            GaussianMarkovRandomFields.loggrad(mean(post), lik)
        @test norm(grad) < 1.0e-8
    end

    @testset "Higher-order models are unaffected" begin
        # Order ≥ 2 models default to CHOLMODFactorization with a sparse precision, so they
        # never hit the carry-over problem. Guard against a regression in the other direction.
        n = 20
        idx = [5, 11]
        A = sparse(1:2, idx, ones(2), 2, n)
        Q_ϵ = Diagonal(fill(80.0, 2))
        y = [0.4, -0.6]

        for prior in (RW2Model(n)(τ = 1.0), ARModel{2}(n)(τ = 1.0, pacf1 = 0.5, pacf2 = 0.2))
            base = prior isa ConstrainedGMRF ? prior.base_gmrf : prior
            @test base.linsolve_cache.alg isa LinearSolve.CHOLMODFactorization

            post = linear_condition(prior; A = A, Q_ϵ = Q_ϵ, y = y)
            post_base = post isa ConstrainedGMRF ? post.base_gmrf : post
            @test post_base.linsolve_cache.alg isa LinearSolve.CHOLMODFactorization
            @test all(isfinite, mean(post))
        end
    end
end
