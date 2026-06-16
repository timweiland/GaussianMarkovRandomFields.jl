using GaussianMarkovRandomFields
using Distributions: logpdf, Normal, NegativeBinomial, Gamma, TDist
using SparseArrays
using LinearAlgebra
using Random

using DifferentiationInterface
using FiniteDiff, ForwardDiff

# Unified `WorkspaceGMRF`-prior IFT tests covering issues #100 (Dual prior)
# and #101 (CompositeLikelihood with Dual hp), plus their combination.
# The single `_workspace_dualhp_ift` helper in
# `ext/forwarddiff/autodiff_likelihood_ift.jl` services every prior×lik
# Dual/Float64 subcase. These tests cross-check the outer-AD gradient
# against finite differences for representative pipelines that downstream
# hyperparameter-inference users would write.

@testset "WorkspaceGMRF dual-hp IFT (issues #100 + #101)" begin
    Random.seed!(42)
    fd_backend = AutoFiniteDiff()

    # ------------------------------------------------------------------------
    # Issue #100: WorkspaceGMRF{Dual} + AutoDiffLikelihood
    # ------------------------------------------------------------------------

    @testset "Dual prior + Float64-hp AutoDiffLikelihood (prior-only Duals)" begin
        # Outer AD over a prior hyperparameter only — lik has no Dual hp.
        # The new dispatch should fire (AutoDiffLikelihood, Dual prior) and
        # thread prior-side partials through the IFT correctly.
        Random.seed!(101)
        k = 8
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)

        function poisson_loglik(x; y, φ)
            return sum(y .* (φ .* x) .- exp.(φ .* x))
        end
        obs_model = AutoDiffObservationModel(
            poisson_loglik;
            n_latent = k,
            hyperparams = (:φ,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )
        # Materialize lik with Float64 hp — it stays Float64 across the pipe.
        obs_lik_primal = obs_model(y; φ = 1.0)

        function pipeline(θ)
            prior = model(ws; τ = exp(θ[1]), ρ = tanh(θ[2]))  # WorkspaceGMRF{Dual}
            posterior = gaussian_approximation(prior, obs_lik_primal)
            return logpdf(posterior, x)
        end

        θ = [0.2, 0.1]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-3
    end

    @testset "Posterior owns the workspace factorization after a Dual GA" begin
        # The forward-mode GA leaves the shared workspace factorized at Q_post
        # (from the IFT tangent solves) and tags the posterior as its owner, so a
        # consumer that touches the posterior first (logpdf / var / logdetcov)
        # reuses that factorization instead of reloading + refactorizing Q_post.
        Random.seed!(202)
        k = 8
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        obs_model = AutoDiffObservationModel(
            (x; y, φ) -> sum(y .* (φ .* x) .- exp.(φ .* x));
            n_latent = k, hyperparams = (:φ,),
            grad_backend = AutoForwardDiff(), hessian_backend = AutoForwardDiff(),
        )
        obs_lik = obs_model(y; φ = 1.0)

        Tag = typeof(ForwardDiff.Tag(:handoff, Float64))
        prior = model(
            ws;
            τ = exp(ForwardDiff.Dual{Tag}(0.1, 1.0, 0.0)),
            ρ = tanh(ForwardDiff.Dual{Tag}(0.2, 0.0, 1.0)),
        )  # WorkspaceGMRF{Dual}
        posterior = gaussian_approximation(prior, obs_lik)

        @test posterior.workspace.loaded_version == posterior.version
        @test posterior.workspace.numeric_valid
        # The reused-factor path stays consistent (finite, right Dual tag).
        @test isfinite(ForwardDiff.value(logpdf(posterior, zeros(k))))
    end

    @testset "Dual prior + Dual-hp AutoDiffLikelihood (joint outer AD)" begin
        # Both prior precision/mean and lik hyperparam carry Duals from the
        # same outer ForwardDiff pass over θ = (τ, φ). This is the canonical
        # #100 case: a compositional latent model on top of a non-Gaussian
        # likelihood, with outer-AD over all hyperparameters.
        Random.seed!(102)
        k = 8
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)

        function poisson_loglik(x; y, φ)
            return sum(y .* (φ .* x) .- exp.(φ .* x))
        end
        obs_model = AutoDiffObservationModel(
            poisson_loglik;
            n_latent = k,
            hyperparams = (:φ,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )

        function pipeline(θ)
            prior = model(ws; τ = exp(θ[1]), ρ = tanh(θ[2]))     # WorkspaceGMRF{Dual}
            obs_lik = obs_model(y; φ = exp(θ[3]))                # Dual hp
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x)
        end

        θ = [0.0, 0.1, 0.0]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-3
    end

    @testset "Constrained Dual prior + Dual-hp AutoDiffLikelihood" begin
        # RW1 sum-to-zero prior with Dual τ + Gaussian loglik with Dual α.
        # Exercises the constrained branch with simultaneous prior + lik
        # Duals — the most demanding IFT subcase.
        Random.seed!(103)
        k = 8
        y = randn(k) .* 0.3
        x_eval = randn(k); x_eval .-= sum(x_eval) / k
        model = RW1Model(k)
        ws = make_workspace(model; τ = 1.0)

        function gauss_α(x; y, α)
            return -0.5 * α * sum((y .- x) .^ 2) +
                0.5 * length(y) * log(α)
        end
        obs_model = AutoDiffObservationModel(
            gauss_α;
            n_latent = k,
            hyperparams = (:α,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )

        function pipeline(θ)
            prior = model(ws; τ = exp(θ[1]))      # constrained WorkspaceGMRF{Dual}
            obs_lik = obs_model(y; α = exp(θ[2])) # Dual hp
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [0.1, log(2.0)]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-3
    end

    # ------------------------------------------------------------------------
    # Issue #101: CompositeLikelihood with Dual-hp AutoDiffLikelihood components
    # ------------------------------------------------------------------------

    @testset "Float64 prior + CompositeLikelihood with two Dual-hp channels" begin
        # Two AutoDiffLikelihood Gaussian channels with distinct outer hp
        # routed via per-component kwargs (the #96 routing feature). The new
        # composite dispatch should detect the Dual components and drive the
        # unified IFT path.
        Random.seed!(104)
        k = 8
        y_a = randn(k) .* 0.3
        y_b = randn(k) .* 0.5 .+ 0.1
        x_eval = randn(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        prior = model(ws; τ = 1.0, ρ = 0.3)  # WorkspaceGMRF{Float64}

        function gauss_loglik(x; y, σ)
            return -0.5 * sum((y .- x) .^ 2) / σ^2 - length(y) * log(σ)
        end
        function gauss_pointwise(x; y, σ)
            return [-0.5 * (y[i] - x[i])^2 / σ^2 - log(σ) for i in eachindex(y)]
        end
        m1 = AutoDiffObservationModel(
            gauss_loglik;
            n_latent = k,
            hyperparams = (:σ,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
            pointwise_loglik_func = gauss_pointwise,
        )
        m2 = AutoDiffObservationModel(
            gauss_loglik;
            n_latent = k,
            hyperparams = (:σ,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
            pointwise_loglik_func = gauss_pointwise,
        )
        composite = CompositeObservationModel(
            (m1, m2),
            ((σ = :σ_a,), (σ = :σ_b,)),
        )

        function pipeline(θ)
            obs_lik = composite(
                CompositeObservations((y_a, y_b));
                σ_a = exp(θ[1]), σ_b = exp(θ[2]),
            )
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [log(0.3), log(0.5)]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-3
    end

    @testset "Float64 prior + CompositeLikelihood: only one Dual component" begin
        # Mixed composite — one Dual-hp Gaussian + one Float64-hp Poisson.
        # The IFT dispatch should still fire (Dual detected) and the Float64
        # component should pass through cleanly.
        Random.seed!(105)
        k = 8
        y_g = randn(k) .* 0.3
        y_p = [2, 1, 3, 0, 4, 1, 2, 3]
        x_eval = zeros(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        prior = model(ws; τ = 1.0, ρ = 0.3)

        function gauss_loglik(x; y, σ)
            return -0.5 * sum((y .- x) .^ 2) / σ^2 - length(y) * log(σ)
        end
        function poisson_loglik(x; y)
            return sum(y .* x .- exp.(x))
        end
        m_g = AutoDiffObservationModel(
            gauss_loglik;
            n_latent = k,
            hyperparams = (:σ,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )
        m_p = AutoDiffObservationModel(
            poisson_loglik;
            n_latent = k,
            hyperparams = (),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )
        # Explicit per-component routes: σ goes to m_g only; m_p gets none.
        composite = CompositeObservationModel(
            (m_g, m_p),
            ((σ = :σ,), NamedTuple()),
        )

        function pipeline(θ)
            obs_lik = composite(
                CompositeObservations((y_g, y_p));
                σ = exp(θ[1]),
            )
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [log(0.3)]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-3
    end

    # ------------------------------------------------------------------------
    # Combined: Dual prior + CompositeLikelihood with Dual components
    # ------------------------------------------------------------------------

    @testset "Dual prior + CompositeLikelihood with Dual components" begin
        # The full compositional case from issue #101's "Relationship to the
        # Dual-prior issue" note: outer AD over a prior hyperparameter (τ)
        # AND two channel hyperparameters (σ_a, σ_b). The unified IFT path
        # handles this in one shot.
        Random.seed!(106)
        k = 8
        y_a = randn(k) .* 0.3
        y_b = randn(k) .* 0.5 .+ 0.1
        x_eval = randn(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)

        function gauss_loglik(x; y, σ)
            return -0.5 * sum((y .- x) .^ 2) / σ^2 - length(y) * log(σ)
        end
        m_a = AutoDiffObservationModel(
            gauss_loglik;
            n_latent = k,
            hyperparams = (:σ,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )
        m_b = AutoDiffObservationModel(
            gauss_loglik;
            n_latent = k,
            hyperparams = (:σ,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )
        composite = CompositeObservationModel(
            (m_a, m_b),
            ((σ = :σ_a,), (σ = :σ_b,)),
        )

        function pipeline(θ)
            prior = model(ws; τ = exp(θ[1]), ρ = tanh(θ[2]))
            obs_lik = composite(
                CompositeObservations((y_a, y_b));
                σ_a = exp(θ[3]), σ_b = exp(θ[4]),
            )
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [0.1, 0.0, log(0.3), log(0.5)]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-3
    end

    # ------------------------------------------------------------------------
    # Detection / dispatch unit checks
    # ------------------------------------------------------------------------

    @testset "_lik_carries_dual_hp recognises composite Dual components" begin
        FDExt = Base.get_extension(GaussianMarkovRandomFields, :GaussianMarkovRandomFieldsForwardDiff)
        @assert FDExt !== nothing

        function loglik_simple(x; y, σ)
            return -0.5 * sum((y .- x) .^ 2) / σ^2
        end
        m1 = AutoDiffObservationModel(
            loglik_simple;
            n_latent = 3, hyperparams = (:σ,),
            grad_backend = AutoForwardDiff(), hessian_backend = AutoForwardDiff(),
        )
        m2 = AutoDiffObservationModel(
            loglik_simple;
            n_latent = 3, hyperparams = (:σ,),
            grad_backend = AutoForwardDiff(), hessian_backend = AutoForwardDiff(),
        )
        composite = CompositeObservationModel(
            (m1, m2),
            ((σ = :σ_a,), (σ = :σ_b,)),
        )
        y = ([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])

        # Float64 hp throughout — predicate must be false.
        lik_primal = composite(CompositeObservations(y); σ_a = 1.0, σ_b = 2.0)
        @test FDExt._lik_carries_dual_hp(lik_primal) == false

        # One Dual hp — predicate must be true.
        Tag = ForwardDiff.Tag{Symbol("composite_test"), Float64}
        DualT = ForwardDiff.Dual{Tag, Float64, 1}
        σ_a_dual = DualT(1.0, ForwardDiff.Partials{1, Float64}((1.0,)))
        lik_dual = composite(CompositeObservations(y); σ_a = σ_a_dual, σ_b = 2.0)
        @test FDExt._lik_carries_dual_hp(lik_dual) == true
    end

    @testset "_assemble_q_post_dual: Dual Q_prior + Diagonal H_dual" begin
        # Direct unit test for the new Dual-Q_prior path of the assembly
        # helper. Pattern preservation + correct partial composition.
        FDExt = Base.get_extension(GaussianMarkovRandomFields, :GaussianMarkovRandomFieldsForwardDiff)
        @assert FDExt !== nothing
        Tag = ForwardDiff.Tag{Symbol("test_q_dual_prior"), Float64}
        ForwardDiff.tagcount(Tag)
        DualT = ForwardDiff.Dual{Tag, Float64, 1}

        # Q_prior with Dual nzval (e.g., scaled by Dual τ).
        q_dense = SparseMatrixCSC(spdiagm(-1 => -ones(3), 0 => 2 * ones(4), 1 => -ones(3)))
        nzval_dual = [
            DualT(q_dense.nzval[i], ForwardDiff.Partials{1, Float64}((Float64(i),)))
                for i in eachindex(q_dense.nzval)
        ]
        Q_prior_dual = SparseMatrixCSC(q_dense.m, q_dense.n, q_dense.colptr, q_dense.rowval, nzval_dual)

        h_diag = [
            DualT(-0.5, ForwardDiff.Partials{1, Float64}((-1.0,))),
            DualT(-0.6, ForwardDiff.Partials{1, Float64}((-1.5,))),
            DualT(-0.7, ForwardDiff.Partials{1, Float64}((-2.0,))),
            DualT(-0.8, ForwardDiff.Partials{1, Float64}((-2.5,))),
        ]
        H = Diagonal(h_diag)
        Q_post = FDExt._assemble_q_post_dual(Q_prior_dual, H, DualT, Val(1))

        @test Q_post isa SparseMatrixCSC
        @test Q_post.colptr == Q_prior_dual.colptr
        @test Q_post.rowval == Q_prior_dual.rowval

        # Diagonal entries: Q_prior - h, with partials composed.
        for i in 1:4
            entry = Q_post[i, i]
            q = Q_prior_dual[i, i]
            @test ForwardDiff.value(entry) ≈ ForwardDiff.value(q) - ForwardDiff.value(h_diag[i])
            @test ForwardDiff.partials(entry, 1) ≈
                ForwardDiff.partials(q, 1) - ForwardDiff.partials(h_diag[i], 1)
        end
        # Off-diagonal entries unchanged from Q_prior_dual.
        offdiag = Q_post[1, 2]
        q_offdiag = Q_prior_dual[1, 2]
        @test ForwardDiff.value(offdiag) ≈ ForwardDiff.value(q_offdiag)
        @test ForwardDiff.partials(offdiag, 1) ≈ ForwardDiff.partials(q_offdiag, 1)
    end

    @testset "_assemble_q_post_dual: Dual Q_prior + sparse H pattern subset" begin
        # Same Dual-Q_prior path but with a SparseMatrixCSC H whose pattern
        # is a strict subset of Q_prior's. Pins the second sparse method.
        FDExt = Base.get_extension(GaussianMarkovRandomFields, :GaussianMarkovRandomFieldsForwardDiff)
        Tag = ForwardDiff.Tag{Symbol("test_q_dual_sparse"), Float64}
        ForwardDiff.tagcount(Tag)
        DualT = ForwardDiff.Dual{Tag, Float64, 1}

        q_dense = SparseMatrixCSC(spdiagm(-1 => -ones(3), 0 => 2 * ones(4), 1 => -ones(3)))
        nzval_dual = [
            DualT(q_dense.nzval[i], ForwardDiff.Partials{1, Float64}((Float64(i),)))
                for i in eachindex(q_dense.nzval)
        ]
        Q_prior_dual = SparseMatrixCSC(q_dense.m, q_dense.n, q_dense.colptr, q_dense.rowval, nzval_dual)

        # H is sparse-tridiagonal with a Dual nonzero on (1,2).
        h_off = DualT(0.25, ForwardDiff.Partials{1, Float64}((0.5,)))
        H_sparse = sparse([1, 2], [2, 1], [h_off, h_off], 4, 4)
        Q_post = FDExt._assemble_q_post_dual(Q_prior_dual, H_sparse, DualT, Val(1))
        @test Q_post.colptr == Q_prior_dual.colptr
        @test Q_post.rowval == Q_prior_dual.rowval
        # Q_post[1,2] = Q_prior[1,2] - h_off
        entry = Q_post[1, 2]
        q12 = Q_prior_dual[1, 2]
        @test ForwardDiff.value(entry) ≈ ForwardDiff.value(q12) - ForwardDiff.value(h_off)
        @test ForwardDiff.partials(entry, 1) ≈
            ForwardDiff.partials(q12, 1) - ForwardDiff.partials(h_off, 1)
    end

    @testset "_lik_dual_tag_npartials covers _DualObsLik subtypes" begin
        # Direct unit tests that pin the four `_DualObsLik` overloads
        # (Normal/NegBin/Gamma/StudentT). The IFT path itself doesn't
        # exercise them when the lik is consumed standalone (those go
        # through `_forwarddiff_workspace_ga_obs_dual`), but the unified
        # collector must still recognise them — e.g. when wrapped in a
        # CompositeLikelihood.
        FDExt = Base.get_extension(GaussianMarkovRandomFields, :GaussianMarkovRandomFieldsForwardDiff)
        Tag = ForwardDiff.Tag{Symbol("test_dual_obslik"), Float64}
        ForwardDiff.tagcount(Tag)
        DualT = ForwardDiff.Dual{Tag, Float64, 1}
        d_one = DualT(0.7, ForwardDiff.Partials{1, Float64}((1.0,)))
        d_two = DualT(2.0, ForwardDiff.Partials{1, Float64}((1.0,)))

        normal_lik = ExponentialFamily(Normal)([0.1, 0.2]; σ = d_one)
        @test FDExt._lik_dual_tag_npartials(normal_lik) == (Tag, 1)
        @test FDExt._lik_carries_dual_hp(normal_lik) == true

        negbin_lik = ExponentialFamily(NegativeBinomial)(
            NegativeBinomialObservations([3, 1, 8]); r = d_two,
        )
        @test FDExt._lik_dual_tag_npartials(negbin_lik) == (Tag, 1)
        @test FDExt._lik_carries_dual_hp(negbin_lik) == true

        gamma_lik = ExponentialFamily(Gamma)([0.5, 1.5]; phi = d_one)
        @test FDExt._lik_dual_tag_npartials(gamma_lik) == (Tag, 1)
        @test FDExt._lik_carries_dual_hp(gamma_lik) == true

        studentt_lik = ExponentialFamily(TDist)([0.3, 0.5]; σ = d_one, ν = d_two)
        @test FDExt._lik_dual_tag_npartials(studentt_lik) == (Tag, 1)
        @test FDExt._lik_carries_dual_hp(studentt_lik) == true

        # Default fallback: arbitrary type with no Dual content.
        @test FDExt._lik_dual_tag_npartials("not a likelihood") == (nothing, nothing)
        @test FDExt._lik_carries_dual_hp("not a likelihood") == false
    end

    @testset "Float64 prior + Composite of Dual NormalLikelihood routes through IFT" begin
        # Exercises the CompositeLikelihood dispatch with `_DualObsLik`
        # components (rather than AutoDiffLikelihood). Routes through the
        # unified IFT helper end-to-end.
        Random.seed!(150)
        k = 6
        y_a = randn(k) .* 0.3
        y_b = randn(k) .* 0.5 .+ 0.1
        x_eval = randn(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        prior = model(ws; τ = 1.0, ρ = 0.3)

        m_norm = ExponentialFamily(Normal)
        composite = CompositeObservationModel(
            (m_norm, m_norm),
            ((σ = :σ_a,), (σ = :σ_b,)),
        )

        function pipeline(θ)
            obs_lik = composite(
                CompositeObservations((y_a, y_b));
                σ_a = exp(θ[1]), σ_b = exp(θ[2]),
            )
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [log(0.3), log(0.5)]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-3
    end

    @testset "Float64 prior + Composite of LTL-wrapped Dual NormalLikelihood (issue #110)" begin
        # The unified IFT helpers must recurse through `LinearlyTransformedLikelihood`.
        # Each composite component is `LTL(NormalLikelihood{Dual})`; without LTL
        # recursion in `_lik_carries_dual_hp` / `_primal_obs_lik` the dispatch
        # falls back to the primal Newton path and dies on a Dual Hessian
        # `setindex!` into the Float64 workspace.
        Random.seed!(160)
        n_latent = 3
        k_phys, k_data = 8, 5
        A_phys = sparse(randn(k_phys, n_latent))
        A_data = sparse(randn(k_data, n_latent))
        y_phys = randn(k_phys)
        y_data = randn(k_data)
        x_eval = randn(n_latent)

        # Prior Q must have a sparsity pattern that subsumes the LTL-induced
        # `A' diag(...) A` Hessian (dense `n_latent × n_latent`). A diagonal
        # prior would trip the `_sparse_hessian_map` pattern check before the
        # IFT recursion ever runs.
        Q_prior = sparse(2.0I + 0.1 * (ones(n_latent, n_latent) - I))
        ws = GMRFWorkspace(Q_prior)
        prior = WorkspaceGMRF(zeros(n_latent), Q_prior, ws)

        m_norm = ExponentialFamily(Normal)
        components = (
            LinearlyTransformedObservationModel(m_norm, A_phys),
            LinearlyTransformedObservationModel(m_norm, A_data),
        )
        composite = CompositeObservationModel(
            components,
            ((σ = :σ_phys,), (σ = :σ_data,)),
        )
        y_composite = CompositeObservations((y_phys, y_data))

        function pipeline(θ)
            obs_lik = composite(y_composite; σ_phys = exp(θ[1]), σ_data = exp(θ[2]))
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [log(0.3), log(1.2)]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test all(isfinite, grad_fwd)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-3
    end

    @testset "Tag-mismatch error paths" begin
        # The IFT collectors guard against partials from independent outer
        # ForwardDiff passes silently mixing. Verify the three error
        # surfaces: (1) prior-vs-lik mismatch, (2) composite-component
        # mismatch, (3) "no Dual partials" sentinel inside the helper.
        FDExt = Base.get_extension(GaussianMarkovRandomFields, :GaussianMarkovRandomFieldsForwardDiff)

        TagA = ForwardDiff.Tag{Symbol("ift_TagA"), Float64}
        TagB = ForwardDiff.Tag{Symbol("ift_TagB"), Float64}
        ForwardDiff.tagcount(TagA); ForwardDiff.tagcount(TagB)

        # (1) Prior carries TagA, lik carries TagB → error.
        DualA = ForwardDiff.Dual{TagA, Float64, 1}
        DualB = ForwardDiff.Dual{TagB, Float64, 1}

        k = 4
        Q_dense = SparseMatrixCSC(spdiagm(0 => 2 * ones(k)))
        Q_dual_A = SparseMatrixCSC(
            Q_dense.m, Q_dense.n, Q_dense.colptr, Q_dense.rowval,
            [DualA(Q_dense.nzval[i], ForwardDiff.Partials{1, Float64}((1.0,))) for i in eachindex(Q_dense.nzval)],
        )
        μ_A = [DualA(0.0, ForwardDiff.Partials{1, Float64}((0.0,))) for _ in 1:k]
        ws_A = GMRFWorkspace(Q_dense)
        prior_A = WorkspaceGMRF(μ_A, Q_dual_A, ws_A)

        function lik_loglik(x; y, σ)
            return -0.5 * sum((y .- x) .^ 2) / σ^2
        end
        obs_model = AutoDiffObservationModel(
            lik_loglik;
            n_latent = k, hyperparams = (:σ,),
            grad_backend = AutoForwardDiff(), hessian_backend = AutoForwardDiff(),
        )
        σ_B = DualB(1.0, ForwardDiff.Partials{1, Float64}((1.0,)))
        lik_B = obs_model(zeros(k); σ = σ_B)
        @test_throws ArgumentError gaussian_approximation(prior_A, lik_B)

        # (2) Composite components carry mismatched tags → error.
        m1 = AutoDiffObservationModel(
            lik_loglik; n_latent = k, hyperparams = (:σ,),
            grad_backend = AutoForwardDiff(), hessian_backend = AutoForwardDiff(),
        )
        m2 = AutoDiffObservationModel(
            lik_loglik; n_latent = k, hyperparams = (:σ,),
            grad_backend = AutoForwardDiff(), hessian_backend = AutoForwardDiff(),
        )
        comp_model = CompositeObservationModel(
            (m1, m2), ((σ = :σ_a,), (σ = :σ_b,))
        )
        σ_a_A = DualA(1.0, ForwardDiff.Partials{1, Float64}((1.0,)))
        comp_lik_mismatched = comp_model(
            CompositeObservations((zeros(k), zeros(k)));
            σ_a = σ_a_A, σ_b = σ_B,
        )
        @test_throws ArgumentError FDExt._lik_dual_tag_npartials(comp_lik_mismatched)

        # (3) No-Duals sentinel: helper called directly with a Float64
        # prior + Float64-hp lik should error before doing IFT work.
        prior_float = WorkspaceGMRF(zeros(3), spdiagm(0 => ones(3)))
        lik_primal = obs_model(zeros(k); σ = 1.0)
        @test_throws ArgumentError FDExt._workspace_dualhp_ift(prior_float, lik_primal)
    end
end
