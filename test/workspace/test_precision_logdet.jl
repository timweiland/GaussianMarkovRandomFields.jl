using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: has_constraints
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using ForwardDiff

const GMRFs = GaussianMarkovRandomFields

@testset "precision_logdet structure hook" begin
    Random.seed!(20260810)

    n_t, n_s = 6, 8
    N = n_t * n_s
    tm = AR1Model(n_t)
    sm = AR1Model(n_s)
    sep = SeparableModel(tm, sm)
    θ = (τ_ar1 = 1.2, ρ_ar1 = 0.6, τ_ar1_2 = 0.9, ρ_ar1_2 = 0.4)
    v0 = [1.2, 0.6, 0.9, 0.4]

    Qj(v) = kron(
        sparse(precision_matrix(tm; τ = v[1], ρ = v[2])),
        sparse(precision_matrix(sm; τ = v[3], ρ = v[4])),
    )
    Q_ref = Qj(v0)

    @testset "Hook values" begin
        # Kronecker factor rule
        @test precision_logdet(sep; θ...) ≈ logdet(Matrix(Q_ref))

        # Closed forms
        @test precision_logdet(IIDModel(7); τ = 2.5) ≈ 7 * log(2.5)
        @test precision_logdet(FixedEffectsModel(3; λ = 0.1)) ≈ 3 * log(0.1)

        # No hook → nothing (callers factorize as before)
        @test precision_logdet(AR1Model(5); τ = 1.0, ρ = 0.5) === nothing

        # CombinedModel: sum when every component provides a value...
        comb = CombinedModel(sep, IIDModel(5))
        θc = (
            τ_ar1_separable = 1.2, ρ_ar1_separable = 0.6, τ_ar1_2_separable = 0.9,
            ρ_ar1_2_separable = 0.4, τ_iid = 2.0,
        )
        Qc_ref = blockdiag(Q_ref, sparse(2.0 * I, 5, 5))
        @test precision_logdet(comb; θc...) ≈ logdet(Matrix(Qc_ref))
        # ...and nothing when any component lacks one
        comb_mixed = CombinedModel(IIDModel(4), AR1Model(3))
        @test precision_logdet(comb_mixed; τ_iid = 1.0, τ_ar1 = 1.0, ρ_ar1 = 0.5) === nothing

        # A single constrained component keeps the pure Kronecker product
        sep_con = SeparableModel(RW1Model(n_t), sm)
        θr = (τ_rw1 = 1.0, τ_ar1 = 0.9, ρ_ar1 = 0.4)
        Qr = kron(
            sparse(precision_matrix(RW1Model(n_t); τ = 1.0)),
            sparse(precision_matrix(sm; τ = 0.9, ρ = 0.4)),
        )
        @test precision_logdet(sep_con; θr...) ≈ logdet(Matrix(Qr))

        # ≥2 constrained components: the joint εI regularization means the
        # factor rule no longer describes the materialized precision
        @test precision_logdet(SeparableModel(RW1Model(5), RW1Model(6)); τ_rw1 = 1.0, τ_rw1_2 = 2.0) === nothing
    end

    @testset "Workspace priors carry the hook value" begin
        ws = make_workspace(sep; θ...)
        prior = sep(ws; θ...)
        @test prior isa WorkspaceGMRF
        @test prior.precision_logdet !== nothing
        @test prior.precision_logdet ≈ logdet(Matrix(Q_ref))

        # logdetcov answers from the field without factorizing the workspace
        @test !ws.numeric_valid  # materialization marked values stale
        @test logdetcov(prior) ≈ -logdet(Matrix(Q_ref))
        @test !ws.numeric_valid  # ...and the logdet did not factorize

        z = randn(N)
        @test logpdf(prior, z) ≈ logpdf(sep(; θ...), z) rtol = 1.0e-8

        # Constrained prior: hook value present, Rue–Held correction unchanged
        sep_con = SeparableModel(RW1Model(n_t), sm)
        θr = (τ_rw1 = 1.0, τ_ar1 = 0.9, ρ_ar1 = 0.4)
        ws3 = make_workspace(sep_con; θr...)
        prior_con = sep_con(ws3; θr...)
        @test has_constraints(prior_con)
        @test prior_con.precision_logdet !== nothing
        xr = randn(N)
        @test logpdf(prior_con, xr) ≈ logpdf(sep_con(; θr...), xr) rtol = 1.0e-8
    end

    @testset "GA and objective equivalence vs hook-less flow" begin
        ws = make_workspace(sep; θ...)
        prior = sep(ws; θ...)
        ws_ref = GMRFWorkspace(Q_ref)
        prior_ref = WorkspaceGMRF(zeros(N), Q_ref, ws_ref)
        @test prior_ref.precision_logdet === nothing

        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations(rand(0:4, N))
        obs_lik = obs_model(y)

        # Objective in the standard evaluation shape (prior logdet before GA)
        function objective(p)
            ldc = logdetcov(p)
            g = gaussian_approximation(p, obs_lik)
            xs = mean(g)
            gaussian_logpdf = logpdf(g, xs)
            r = xs .- mean(p)
            lp_prior = -0.5 * dot(r, precision_matrix(p) * r) - 0.5 * ldc -
                0.5 * length(xs) * log(2π)
            return lp_prior + loglik(xs, obs_lik) - gaussian_logpdf
        end
        @test objective(prior) ≈ objective(prior_ref) rtol = 1.0e-6

        post = gaussian_approximation(prior, obs_lik)
        post_ref = gaussian_approximation(prior_ref, obs_lik)
        @test mean(post) ≈ mean(post_ref) rtol = 1.0e-8
        @test precision_matrix(post) ≈ precision_matrix(post_ref) rtol = 1.0e-8
        # The posterior is workspace-backed as always and carries no override
        @test post.precision_logdet === nothing
    end

    @testset "ForwardDiff hyperparameter gradients" begin
        ws = make_workspace(sep; θ...)
        ws_ref = GMRFWorkspace(Q_ref)

        # logdetcov alone: factor-scale tangents match the dense reference
        ldc(v) = logdetcov(sep(ws; τ_ar1 = v[1], ρ_ar1 = v[2], τ_ar1_2 = v[3], ρ_ar1_2 = v[4]))
        ldc_ref(v) = -logdet(Matrix(Qj(v)))
        @test ForwardDiff.gradient(ldc, v0) ≈ ForwardDiff.gradient(ldc_ref, v0) rtol = 1.0e-8

        # Full Laplace objective: hook flow vs hook-less materialized flow
        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations(rand(0:4, N))
        obs_lik = obs_model(y)
        function obj(v, hooked::Bool)
            θd = (τ_ar1 = v[1], ρ_ar1 = v[2], τ_ar1_2 = v[3], ρ_ar1_2 = v[4])
            p = hooked ? sep(ws; θd...) :
                WorkspaceGMRF(zeros(eltype(v), N), Qj(v), ws_ref)
            g = gaussian_approximation(p, obs_lik)
            xs = mean(g)
            r = xs .- mean(p)
            lp_prior = -0.5 * dot(r, precision_matrix(p) * r) - 0.5 * logdetcov(p) -
                0.5 * length(xs) * log(2π)
            return lp_prior + loglik(xs, obs_lik) - logpdf(g, xs)
        end
        @test obj(v0, true) ≈ obj(v0, false) rtol = 1.0e-8
        g_hook = ForwardDiff.gradient(v -> obj(v, true), v0)
        g_ref = ForwardDiff.gradient(v -> obj(v, false), v0)
        @test g_hook ≈ g_ref rtol = 1.0e-6

        # Observation-hyperparameter gradient with a hook-carrying prior
        obs_model_n = ExponentialFamily(Distributions.Normal)
        yn = randn(N)
        function obj_obs(v, hooked::Bool)
            obs_lik_n = obs_model_n(yn; σ = v[1])
            p = hooked ? sep(ws; θ...) : WorkspaceGMRF(zeros(N), Q_ref, ws_ref)
            g = gaussian_approximation(p, obs_lik_n)
            xs = mean(g)
            return logpdf(p, xs) + loglik(xs, obs_lik_n) - logpdf(g, xs)
        end
        g_obs_hook = ForwardDiff.gradient(v -> obj_obs(v, true), [0.5])
        g_obs_ref = ForwardDiff.gradient(v -> obj_obs(v, false), [0.5])
        @test g_obs_hook ≈ g_obs_ref rtol = 1.0e-5
    end

    @testset "prior_logdensity fast path" begin
        x = randn(N)
        @test prior_logdensity(sep, x; θ...) ≈ logpdf(sep(; θ...), x) rtol = 1.0e-8

        # Dual hyperparameters route through the exact Dual factor logdet
        f(v) = prior_logdensity(sep, x; τ_ar1 = v[1], ρ_ar1 = v[2], τ_ar1_2 = v[3], ρ_ar1_2 = v[4])
        f_ref(v) = 0.5 * logdet(Matrix(Qj(v))) - 0.5 * dot(x, Qj(v), x) - 0.5 * N * log(2π)
        @test f(v0) ≈ f_ref(v0) rtol = 1.0e-8
        @test ForwardDiff.gradient(f, v0) ≈ ForwardDiff.gradient(f_ref, v0) rtol = 1.0e-6

        # Constrained models keep the exact corrected fallback
        sep_con = SeparableModel(RW1Model(n_t), sm)
        θr = (τ_rw1 = 1.0, τ_ar1 = 0.9, ρ_ar1 = 0.4)
        xr = randn(N)
        @test prior_logdensity(sep_con, xr; θr...) ≈ logpdf(sep_con(; θr...), xr) rtol = 1.0e-8
    end

    @testset "Single-component constraints skip redundancy removal" begin
        sep_con = SeparableModel(RW1Model(n_t), sm)
        θr = (τ_rw1 = 1.0, τ_ar1 = 0.9, ρ_ar1 = 0.4)
        A, e = constraints(sep_con; θr...)
        @test size(A, 1) == n_s
        @test Matrix(A) ≈ kron(ones(1, n_t), Matrix(I, n_s, n_s))
        @test rank(Matrix(A)) == n_s
        @test all(iszero, e)

        # Multiple constrained components still go through removal
        A2, e2 = constraints(SeparableModel(RW1Model(3), RW1Model(2)); τ_rw1 = 1.0, τ_rw1_2 = 1.0)
        @test rank(Matrix(A2)) == size(A2, 1)
    end

    @testset "Batched multi-RHS workspace solves" begin
        B = randn(N, 3)
        ws_chol = GMRFWorkspace(Q_ref)
        @test workspace_solve(ws_chol, B) ≈ Matrix(Q_ref) \ B rtol = 1.0e-8
        ws_ct = GMRFWorkspace(Q_ref, CliqueTreesBackend)
        @test workspace_solve(ws_ct, B) ≈ Matrix(Q_ref) \ B rtol = 1.0e-8
    end
end
