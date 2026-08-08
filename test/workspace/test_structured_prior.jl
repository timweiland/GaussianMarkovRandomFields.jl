using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using ForwardDiff
import ChainRulesCore

const GMRFs = GaussianMarkovRandomFields
# Kronecker.jl is a package dependency (not a test extra); reference its
# lazy type through the package namespace.
const AbstractKroneckerProduct = GMRFs.AbstractKroneckerProduct

@testset "Structured prior (Kronecker/blockdiag precisions)" begin
    Random.seed!(20260808)

    n_t, n_s = 6, 8
    time_model = AR1Model(n_t)
    space_model = AR1Model(n_s)
    sep = SeparableModel(time_model, space_model)
    θ = (τ_ar1 = 1.2, ρ_ar1 = 0.6, τ_ar1_2 = 0.9, ρ_ar1_2 = 0.4)

    Q_factor(τt, ρt, τs, ρs) = kron(
        sparse(precision_matrix(time_model; τ = τt, ρ = ρt)),
        sparse(precision_matrix(space_model; τ = τs, ρ = ρs)),
    )
    Q_ref = Q_factor(θ.τ_ar1, θ.ρ_ar1, θ.τ_ar1_2, θ.ρ_ar1_2)

    @testset "precision_matrix return types" begin
        Q = precision_matrix(sep; θ...)
        @test Q isa AbstractKroneckerProduct
        @test GMRFs._ensure_sparse(Q) ≈ Q_ref
        # Lazy type still behaves like the matrix it represents
        @test size(Q) == (n_t * n_s, n_t * n_s)
        v = randn(n_t * n_s)
        @test Q * v ≈ Q_ref * v

        # Single-constrained separable keeps structure (falls back only at
        # workspace instantiation)
        sep_con = SeparableModel(RW1Model(n_t), space_model)
        @test precision_matrix(sep_con; τ_rw1 = 1.0, τ_ar1 = 0.9, ρ_ar1 = 0.4) isa
            AbstractKroneckerProduct

        # ≥2 constrained components: εI regularization destroys structure →
        # materialized sparse, values matching the historical path
        sep2c = SeparableModel(RW1Model(5), RW1Model(6))
        Q2c = precision_matrix(sep2c; τ_rw1 = 1.0, τ_rw1_2 = 2.0)
        @test Q2c isa SparseMatrixCSC
        Q2c_ref = kron(
            sparse(precision_matrix(RW1Model(5); τ = 1.0)),
            sparse(precision_matrix(RW1Model(6); τ = 2.0)),
        )
        reg = max(RW1Model(5).regularization, RW1Model(6).regularization)
        @test Q2c ≈ Q2c_ref + reg * I
    end

    @testset "Workspace instantiation → StructuredPriorGMRF" begin
        ws = make_workspace(sep; θ...)
        prior = sep(ws; θ...)
        @test prior isa StructuredPriorGMRF
        @test prior.precision ≈ Q_ref
        # Snapshot shares the workspace pattern (storage-compatible with the
        # Newton loop's positional copy)
        @test prior.precision.colptr == ws.Q.colptr
        @test ws.prior_cache isa GMRFs.StructuredPriorCache

        # The prior never claims or invalidates the joint workspace
        @test ws.loaded_version == 0

        ld_ref = logdet(Matrix(Q_ref))
        @test logdetcov(prior) ≈ -ld_ref
        x = randn(n_t * n_s)
        @test logpdf(prior, x) ≈
            -0.5 * dot(x, Q_ref, x) + 0.5 * ld_ref - 0.5 * (n_t * n_s) * log(2π)
        @test var(prior) ≈ diag(inv(Matrix(Q_ref)))

        # Cache object is reused across hyperparameter evaluations
        cache1 = ws.prior_cache
        prior2 = sep(ws; τ_ar1 = 2.0, ρ_ar1 = 0.3, τ_ar1_2 = 1.1, ρ_ar1_2 = 0.2)
        @test ws.prior_cache === cache1
        @test logdetcov(prior2) ≈ -logdet(Matrix(Q_factor(2.0, 0.3, 1.1, 0.2)))

        # Constrained separable prior falls back to the materialized path
        sep_con = SeparableModel(RW1Model(n_t), space_model)
        θr = (τ_rw1 = 1.0, τ_ar1 = 0.9, ρ_ar1 = 0.4)
        ws3 = make_workspace(sep_con; θr...)
        prior_con = sep_con(ws3; θr...)
        @test prior_con isa WorkspaceGMRF
        @test has_constraints(prior_con)
    end

    @testset "Gaussian approximation matches materialized prior flow" begin
        ws = make_workspace(sep; θ...)
        prior = sep(ws; θ...)

        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations(rand(0:4, n_t * n_s))
        obs_lik = obs_model(y)

        post = gaussian_approximation(prior, obs_lik)
        @test post isa WorkspaceGMRF

        ws_ref = GMRFWorkspace(Q_ref)
        prior_ref = WorkspaceGMRF(zeros(n_t * n_s), Q_ref, ws_ref)
        post_ref = gaussian_approximation(prior_ref, obs_lik)

        @test mean(post) ≈ mean(post_ref) rtol = 1.0e-8
        @test precision_matrix(post) ≈ precision_matrix(post_ref) rtol = 1.0e-8
        @test logpdf(post, mean(post)) ≈ logpdf(post_ref, mean(post_ref)) rtol = 1.0e-6

        # Full Laplace-objective assembly (Latte's evaluation shape)
        function objective(prior_g, post_g)
            xs = mean(post_g)
            r = xs .- mean(prior_g)
            lp_prior = -0.5 * dot(r, precision_matrix(prior_g) * r) -
                0.5 * logdetcov(prior_g) - 0.5 * length(xs) * log(2π)
            return lp_prior + loglik(xs, obs_lik) - logpdf(post_g, xs)
        end
        @test objective(prior, post) ≈ objective(prior_ref, post_ref) rtol = 1.0e-6
    end

    @testset "CombinedModel composition (BlockDiagonalPrecision)" begin
        iid = IIDModel(5)
        comb = CombinedModel(sep, iid)
        θc = (
            τ_ar1_separable = 1.2, ρ_ar1_separable = 0.6, τ_ar1_2_separable = 0.9,
            ρ_ar1_2_separable = 0.4, τ_iid = 2.0,
        )
        Qc = precision_matrix(comb; θc...)
        @test Qc isa BlockDiagonalPrecision
        Qc_ref = blockdiag(Q_ref, sparse(2.0 * I, 5, 5))
        @test GMRFs._ensure_sparse(Qc) ≈ Qc_ref
        v = randn(size(Qc_ref, 1))
        @test Qc * v ≈ Qc_ref * v

        wsc = make_workspace(comb; θc...)
        prior_c = comb(wsc; θc...)
        @test prior_c isa StructuredPriorGMRF
        @test logdetcov(prior_c) ≈ -logdet(Matrix(Qc_ref))
        @test var(prior_c) ≈ diag(inv(Matrix(Qc_ref)))

        # Sparse-only CombinedModel keeps the materialized blockdiag path
        comb_sparse = CombinedModel(AR1Model(4), IIDModel(3))
        Qs = precision_matrix(comb_sparse; τ_ar1 = 1.0, ρ_ar1 = 0.5, τ_iid = 1.0)
        @test Qs isa SparseMatrixCSC
    end

    @testset "prior_logdensity fast path" begin
        x = randn(n_t * n_s)
        @test prior_logdensity(sep, x; θ...) ≈ logpdf(sep(; θ...), x) rtol = 1.0e-8

        iid = IIDModel(5)
        comb = CombinedModel(sep, iid)
        θc = (
            τ_ar1_separable = 1.2, ρ_ar1_separable = 0.6, τ_ar1_2_separable = 0.9,
            ρ_ar1_2_separable = 0.4, τ_iid = 2.0,
        )
        xc = randn(length(comb))
        @test prior_logdensity(comb, xc; θc...) ≈ logpdf(comb(; θc...), xc) rtol = 1.0e-8

        # Constrained separable: falls back to the materialized (corrected) density
        sep_con = SeparableModel(RW1Model(n_t), space_model)
        θr = (τ_rw1 = 1.0, τ_ar1 = 0.9, ρ_ar1 = 0.4)
        xr = randn(length(sep_con))
        @test prior_logdensity(sep_con, xr; θr...) ≈ logpdf(sep_con(; θr...), xr) rtol = 1.0e-8
    end

    @testset "ForwardDiff hyperparameter gradients" begin
        ws = make_workspace(sep; θ...)
        ws_ref = GMRFWorkspace(Q_ref)

        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations(rand(0:4, n_t * n_s))
        obs_lik = obs_model(y)

        function obj(v, materialized::Bool)
            T = eltype(v)
            θd = (τ_ar1 = v[1], ρ_ar1 = v[2], τ_ar1_2 = v[3], ρ_ar1_2 = v[4])
            p = if materialized
                Qj = Q_factor(v[1], v[2], v[3], v[4])
                WorkspaceGMRF(zeros(T, n_t * n_s), Qj, ws_ref)
            else
                sep(ws; θd...)
            end
            g = gaussian_approximation(p, obs_lik)
            xs = mean(g)
            r = xs .- mean(p)
            lp_prior = -0.5 * dot(r, precision_matrix(p) * r) - 0.5 * logdetcov(p) -
                0.5 * length(xs) * log(2π)
            return lp_prior + loglik(xs, obs_lik) - logpdf(g, xs)
        end

        θvec = [1.2, 0.6, 0.9, 0.4]
        @test obj(θvec, false) ≈ obj(θvec, true) rtol = 1.0e-8

        g_structured = ForwardDiff.gradient(v -> obj(v, false), θvec)
        g_materialized = ForwardDiff.gradient(v -> obj(v, true), θvec)
        @test g_structured ≈ g_materialized rtol = 1.0e-6

        # Dual-valued logdetcov alone (factor-level selinv tangent)
        ldc(v) = logdetcov(sep(ws; τ_ar1 = v[1], ρ_ar1 = v[2], τ_ar1_2 = v[3], ρ_ar1_2 = v[4]))
        ldc_ref(v) = -logdet(Matrix(Q_factor(v[1], v[2], v[3], v[4])))
        @test ForwardDiff.gradient(ldc, θvec) ≈ ForwardDiff.gradient(ldc_ref, θvec) rtol = 1.0e-8
    end

    @testset "Reverse-mode guards throw" begin
        ws = make_workspace(sep; θ...)
        prior = sep(ws; θ...)
        @test_throws ArgumentError ChainRulesCore.rrule(logdetcov, prior)
        @test_throws ArgumentError ChainRulesCore.rrule(logpdf, prior, randn(length(prior)))
        @test_throws ArgumentError rand(prior)
    end
end
