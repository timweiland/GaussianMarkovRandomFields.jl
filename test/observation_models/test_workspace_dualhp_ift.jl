using GaussianMarkovRandomFields
using Distributions: logpdf
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
end
