using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: has_constraints
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

        # Constrained separable prior stays structured (factor-form Rue–Held)
        sep_con = SeparableModel(RW1Model(n_t), space_model)
        θr = (τ_rw1 = 1.0, τ_ar1 = 0.9, ρ_ar1 = 0.4)
        ws3 = make_workspace(sep_con; θr...)
        prior_con = sep_con(ws3; θr...)
        @test prior_con isa StructuredPriorGMRF
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
    end

    @testset "Sampling (chol(⊗) = ⊗chol)" begin
        sep_u = SeparableModel(AR1Model(4), AR1Model(3))
        θu = (τ_ar1 = 1.2, ρ_ar1 = 0.6, τ_ar1_2 = 0.8, ρ_ar1_2 = 0.3)
        ws_u = make_workspace(sep_u; θu...)
        prior_u = sep_u(ws_u; θu...)
        Qu = Matrix(GMRFs._ensure_sparse(precision_matrix(sep_u; θu...)))
        # The sampling transform M satisfies M Mᵀ = Q⁻¹ exactly — build M by
        # applying it to unit vectors.
        M = hcat(
            [
                GMRFs._structured_sample_transform(
                        Vector{Float64}(I(12)[:, k]), prior_u.structure, prior_u.cache.op
                    ) for k in 1:12
            ]...
        )
        @test M * M' ≈ inv(Qu) rtol = 1.0e-8
        @test all(isfinite, rand(prior_u))

        comb_u = CombinedModel(sep_u, IIDModel(3))
        θcu = (
            τ_ar1_separable = 1.2, ρ_ar1_separable = 0.6, τ_ar1_2_separable = 0.8,
            ρ_ar1_2_separable = 0.3, τ_iid = 2.0,
        )
        ws_cu = make_workspace(comb_u; θcu...)
        prior_cu = comb_u(ws_cu; θcu...)
        Qcu = Matrix(GMRFs._ensure_sparse(precision_matrix(comb_u; θcu...)))
        Mc = hcat(
            [
                GMRFs._structured_sample_transform(
                        Vector{Float64}(I(15)[:, k]), prior_cu.structure, prior_cu.cache.op
                    ) for k in 1:15
            ]...
        )
        @test Mc * Mc' ≈ inv(Qcu) rtol = 1.0e-8
    end
end

@testset "Structured prior — constraints (factor-form Rue–Held)" begin
    Random.seed!(20260809)

    n_t, n_s = 6, 8
    N = n_t * n_s
    rw = RW1Model(n_t)
    space_model = AR1Model(n_s)
    sep = SeparableModel(rw, space_model)
    θ = (τ_rw1 = 1.3, τ_ar1 = 0.9, ρ_ar1 = 0.4)

    Q_ref = kron(
        sparse(precision_matrix(rw; τ = θ.τ_rw1)),
        sparse(precision_matrix(space_model; τ = θ.τ_ar1, ρ = θ.ρ_ar1)),
    )

    @testset "Constraint detection" begin
        kc = GMRFs._prior_constraints(sep; θ...)
        @test kc isa GMRFs.KroneckerConstraint
        A_ref, e_ref = constraints(sep; θ...)
        @test kc.A ≈ sparse(A_ref)
        @test kc.e ≈ e_ref

        # ≥2 constrained components: falls back to the general constraint path
        sep2c = SeparableModel(RW1Model(5), RW1Model(6))
        @test !(GMRFs._prior_constraints(sep2c; τ_rw1 = 1.0, τ_rw1_2 = 2.0) isa GMRFs.KroneckerConstraint)
    end

    @testset "Constrained prior vs ConstraintInfo reference" begin
        kc = GMRFs._prior_constraints(sep; θ...)
        ws = make_workspace(sep; θ...)
        prior = sep(ws; θ...)
        @test prior isa StructuredPriorGMRF && has_constraints(prior)
        @test ws.loaded_version == 0  # prior never claimed the joint workspace

        ws_ref = GMRFWorkspace(Q_ref)
        prior_ref = GMRFs._instantiate_prior(Q_ref, zeros(N), (kc.A, kc.e), ws_ref)

        # Compare at a constraint-satisfying point
        x = randn(N)
        Ad = Matrix(kc.A)
        x_c = x - Ad' * ((Ad * Ad') \ (Ad * x))
        @test logpdf(prior, x_c) ≈ logpdf(prior_ref, x_c) rtol = 1.0e-10
        @test mean(prior) ≈ mean(prior_ref)
        @test var(prior) ≈ var(prior_ref) rtol = 1.0e-8
        # Base logdetcov excludes the Rue–Held correction (same as WorkspaceGMRF)
        @test logdetcov(prior) ≈ -logdet(Matrix(Q_ref))
        @test_throws ArgumentError rand(prior)
    end

    @testset "Constrained GA + ForwardDiff gradient vs materialized flow" begin
        kc = GMRFs._prior_constraints(sep; θ...)
        ws = make_workspace(sep; θ...)
        ws_ref = GMRFWorkspace(Q_ref)

        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations(rand(0:4, N))
        obs_lik = obs_model(y)

        prior = sep(ws; θ...)
        prior_ref = GMRFs._instantiate_prior(Q_ref, zeros(N), (kc.A, kc.e), ws_ref)
        post = gaussian_approximation(prior, obs_lik)
        post_ref = gaussian_approximation(prior_ref, obs_lik)
        @test post isa WorkspaceGMRF && has_constraints(post)
        @test mean(post) ≈ mean(post_ref) rtol = 1.0e-6
        @test precision_matrix(post) ≈ precision_matrix(post_ref)
        @test norm(kc.A * mean(post)) < 1.0e-6

        # Latte-shaped constrained objective (logpdf-based prior term)
        function obj(v, structured::Bool)
            θd = (τ_rw1 = v[1], τ_ar1 = v[2], ρ_ar1 = v[3])
            p = if structured
                sep(ws; θd...)
            else
                T = eltype(v)
                Qj = kron(
                    sparse(precision_matrix(rw; τ = v[1])),
                    sparse(precision_matrix(space_model; τ = v[2], ρ = v[3])),
                )
                GMRFs._instantiate_prior(Qj, zeros(T, N), (kc.A, kc.e), ws_ref)
            end
            g = gaussian_approximation(p, obs_lik)
            xs = mean(g)
            return logpdf(p, xs) + loglik(xs, obs_lik) - logpdf(g, xs)
        end
        v0 = [1.3, 0.9, 0.4]
        @test obj(v0, true) ≈ obj(v0, false) rtol = 1.0e-8
        g_s = ForwardDiff.gradient(v -> obj(v, true), v0)
        g_m = ForwardDiff.gradient(v -> obj(v, false), v0)
        @test g_s ≈ g_m rtol = 1.0e-6
    end

    @testset "CombinedModel block embedding" begin
        iid = IIDModel(5)
        comb = CombinedModel(sep, iid)
        θc = (
            τ_rw1_separable = 1.3, τ_ar1_separable = 0.9, ρ_ar1_separable = 0.4,
            τ_iid = 2.0,
        )
        kcc = GMRFs._prior_constraints(comb; θc...)
        @test kcc isa GMRFs.KroneckerConstraint
        Ac_ref, ec_ref = constraints(comb; θc...)
        @test Matrix(kcc.A) ≈ Matrix(Ac_ref)

        wsc = make_workspace(comb; θc...)
        prior_c = comb(wsc; θc...)
        @test prior_c isa StructuredPriorGMRF && has_constraints(prior_c)

        Qc_ref = blockdiag(Q_ref, sparse(2.0 * I, 5, 5))
        wsc_ref = GMRFWorkspace(Qc_ref)
        prior_c_ref = GMRFs._instantiate_prior(Qc_ref, zeros(N + 5), (kcc.A, kcc.e), wsc_ref)
        xc = randn(N + 5)
        Acd = Matrix(kcc.A)
        xc_c = xc - Acd' * ((Acd * Acd') \ (Acd * xc))
        @test logpdf(prior_c, xc_c) ≈ logpdf(prior_c_ref, xc_c) rtol = 1.0e-10
        @test var(prior_c) ≈ var(prior_c_ref) rtol = 1.0e-8
    end

    @testset "Observation-hyperparameter gradients (Dual likelihood)" begin
        obs_model = ExponentialFamily(Distributions.Normal)
        y = randn(N)
        ws = make_workspace(sep; θ...)
        ws_ref = GMRFWorkspace(Q_ref)
        kc = GMRFs._prior_constraints(sep; θ...)

        function obj_obs(v, structured::Bool)
            obs_lik = obs_model(y; σ = v[1])
            p = if structured
                sep(ws; θ...)
            else
                GMRFs._instantiate_prior(Q_ref, zeros(N), (kc.A, kc.e), ws_ref)
            end
            g = gaussian_approximation(p, obs_lik)
            xs = mean(g)
            return logpdf(p, xs) + loglik(xs, obs_lik) - logpdf(g, xs)
        end
        σ0 = [0.5]
        @test obj_obs(σ0, true) ≈ obj_obs(σ0, false) rtol = 1.0e-6
        g_s = ForwardDiff.gradient(v -> obj_obs(v, true), σ0)
        g_m = ForwardDiff.gradient(v -> obj_obs(v, false), σ0)
        @test g_s ≈ g_m rtol = 1.0e-5

        # Unconstrained variant (covers the unconstrained obs-Dual branch)
        sep_u = SeparableModel(AR1Model(n_t), space_model)
        θu = (τ_ar1 = 1.2, ρ_ar1 = 0.6, τ_ar1_2 = 0.9, ρ_ar1_2 = 0.4)
        ws_u = make_workspace(sep_u; θu...)
        Q_u = GMRFs._ensure_sparse(precision_matrix(sep_u; θu...))
        ws_u_ref = GMRFWorkspace(Q_u)
        function obj_obs_u(v, structured::Bool)
            obs_lik = obs_model(y; σ = v[1])
            p = structured ? sep_u(ws_u; θu...) :
                WorkspaceGMRF(zeros(N), Q_u, ws_u_ref)
            g = gaussian_approximation(p, obs_lik)
            xs = mean(g)
            return logpdf(p, xs) + loglik(xs, obs_lik) - logpdf(g, xs)
        end
        g_su = ForwardDiff.gradient(v -> obj_obs_u(v, true), σ0)
        g_mu = ForwardDiff.gradient(v -> obj_obs_u(v, false), σ0)
        @test g_su ≈ g_mu rtol = 1.0e-5
    end

    @testset "Dual prior_logdensity (cacheless factor engines)" begin
        sep_u = SeparableModel(AR1Model(n_t), space_model)
        x = randn(N)
        f(v) = prior_logdensity(sep_u, x; τ_ar1 = v[1], ρ_ar1 = v[2], τ_ar1_2 = v[3], ρ_ar1_2 = v[4])
        function f_ref(v)
            Qj = kron(
                sparse(precision_matrix(AR1Model(n_t); τ = v[1], ρ = v[2])),
                sparse(precision_matrix(space_model; τ = v[3], ρ = v[4])),
            )
            return 0.5 * logdet(Matrix(Qj)) - 0.5 * dot(x, Qj, x) - 0.5 * N * log(2π)
        end
        v0 = [1.2, 0.6, 0.9, 0.4]
        @test f(v0) ≈ f_ref(v0) rtol = 1.0e-8
        @test ForwardDiff.gradient(f, v0) ≈ ForwardDiff.gradient(f_ref, v0) rtol = 1.0e-6
    end

    @testset "Batched multi-RHS solve (generic backend fallback)" begin
        ws_ct = GMRFWorkspace(Q_ref, CliqueTreesBackend)
        B = randn(N, 3)
        @test workspace_solve(ws_ct, B) ≈ Matrix(Q_ref) \ B rtol = 1.0e-8
    end

    @testset "Fallback instantiation paths" begin
        # Structured precision + general (tuple) constraints: two constrained
        # components force the dense constraint assembly, and the prior
        # materializes.
        W = sparse([0 1 0; 1 0 1; 0 1 0.0])
        comb2 = CombinedModel(sep, BesagModel(W))
        θ2c = (
            τ_rw1_separable = 1.3, τ_ar1_separable = 0.9, ρ_ar1_separable = 0.4,
            τ_besag = 1.0,
        )
        @test precision_matrix(comb2; θ2c...) isa BlockDiagonalPrecision
        @test !(GMRFs._prior_constraints(comb2; θ2c...) isa GMRFs.KroneckerConstraint)
        ws2c = make_workspace(comb2; θ2c...)
        prior2c = comb2(ws2c; θ2c...)
        @test prior2c isa WorkspaceGMRF && has_constraints(prior2c)

        # Non-homogeneous structured constraint (e ≠ 0): falls back to the
        # materialized path rather than resolving factor-form corrections.
        kc = GMRFs._prior_constraints(sep; θ...)
        kc_inhom = GMRFs.KroneckerConstraint(kc.A, ones(length(kc.e)), kc.comp, kc.A_i, kc.block)
        ws_f = make_workspace(sep; θ...)
        Q_lazy = precision_matrix(sep; θ...)
        prior_f = GMRFs._instantiate_prior(Q_lazy, zeros(N), kc_inhom, ws_f)
        @test prior_f isa WorkspaceGMRF && has_constraints(prior_f)
    end

    @testset "Diagonal constrained factor (constrained IID in a Kronecker product)" begin
        iid_con = IIDModel(4; constraint = :sumtozero)
        sep_dc = SeparableModel(iid_con, AR1Model(5))
        θdc = (τ_iid = 2.0, τ_ar1 = 1.1, ρ_ar1 = 0.3)
        kc = GMRFs._prior_constraints(sep_dc; θdc...)
        @test kc isa GMRFs.KroneckerConstraint
        ws_dc = make_workspace(sep_dc; θdc...)
        prior_dc = sep_dc(ws_dc; θdc...)
        @test prior_dc isa StructuredPriorGMRF && has_constraints(prior_dc)

        Q_dc = kron(
            sparse(precision_matrix(iid_con; τ = 2.0)),
            sparse(precision_matrix(AR1Model(5); τ = 1.1, ρ = 0.3)),
        )
        ws_dc_ref = GMRFWorkspace(Q_dc)
        prior_dc_ref = GMRFs._instantiate_prior(Q_dc, zeros(20), (kc.A, kc.e), ws_dc_ref)
        xd = randn(20)
        Ad = Matrix(kc.A)
        xd_c = xd - Ad' * ((Ad * Ad') \ (Ad * xd))
        @test logpdf(prior_dc, xd_c) ≈ logpdf(prior_dc_ref, xd_c) rtol = 1.0e-10
        @test var(prior_dc) ≈ var(prior_dc_ref) rtol = 1.0e-8
    end

    @testset "Display, std, and guard methods" begin
        ws = make_workspace(sep; θ...)
        prior = sep(ws; θ...)
        @test occursin("⊗", sprint(show, prior))
        @test occursin("constraints", sprint(show, prior))
        @test std(prior) ≈ sqrt.(var(prior))

        obs_model = ExponentialFamily(Distributions.Poisson)
        obs_lik = obs_model(PoissonObservations(rand(0:4, N)))
        @test_throws ArgumentError ChainRulesCore.rrule(gaussian_approximation, prior, obs_lik)
        @test_throws ArgumentError GMRFs._workspace_add_precision_tangent(nothing, prior, nothing)

        # BlockDiagonalPrecision basics
        B = BlockDiagonalPrecision(sparse(2.0 * I, 3, 3), Diagonal([1.0, 2.0]))
        @test B[1, 5] == 0.0
        @test B[5, 5] == 2.0
        @test_throws ArgumentError BlockDiagonalPrecision(randn(2, 3))
        combu = CombinedModel(SeparableModel(AR1Model(3), AR1Model(2)), IIDModel(2))
        Qcu = precision_matrix(
            combu;
            τ_ar1_separable = 1.0, ρ_ar1_separable = 0.5,
            τ_ar1_2_separable = 1.0, ρ_ar1_2_separable = 0.3, τ_iid = 1.0,
        )
        @test occursin("blockdiag", GMRFs._structure_summary(Qcu))
    end

    @testset "Multi-row component constraints (RW2)" begin
        rw2 = RW2Model(7)
        sep2 = SeparableModel(rw2, AR1Model(4))
        θ2 = (τ_rw2 = 1.0, τ_ar1 = 1.1, ρ_ar1 = 0.3)
        kc2 = GMRFs._prior_constraints(sep2; θ2...)
        @test kc2 isa GMRFs.KroneckerConstraint
        @test size(kc2.A_i, 1) == 2

        ws4 = make_workspace(sep2; θ2...)
        prior2 = sep2(ws4; θ2...)
        @test prior2 isa StructuredPriorGMRF && has_constraints(prior2)
        Q2_ref = kron(
            sparse(precision_matrix(rw2; τ = 1.0)),
            sparse(precision_matrix(AR1Model(4); τ = 1.1, ρ = 0.3)),
        )
        ws4_ref = GMRFWorkspace(Q2_ref)
        prior2_ref = GMRFs._instantiate_prior(Q2_ref, zeros(28), (kc2.A, kc2.e), ws4_ref)
        x2 = randn(28)
        A2d = Matrix(kc2.A)
        x2_c = x2 - A2d' * ((A2d * A2d') \ (A2d * x2))
        @test logpdf(prior2, x2_c) ≈ logpdf(prior2_ref, x2_c) rtol = 1.0e-8
        @test var(prior2) ≈ var(prior2_ref) rtol = 1.0e-6
    end
end
