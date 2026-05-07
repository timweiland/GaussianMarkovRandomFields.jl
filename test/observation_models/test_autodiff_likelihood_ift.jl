using GaussianMarkovRandomFields
using Distributions: logpdf
using SparseArrays
using LinearAlgebra
using Random

using DifferentiationInterface
using FiniteDiff, ForwardDiff

# IFT-path tests for `AutoDiffLikelihood` with Dual-valued hyperparameters.
#
# The path is implemented in `ext/forwarddiff/autodiff_likelihood_ift.jl`
# and dispatched when `gaussian_approximation` is called with a Float64
# `WorkspaceGMRF` prior + an `AutoDiffLikelihood` whose hyperparams carry
# AD partials. The workspace prior is a precondition for the dispatch but
# the tests here are about the IFT machinery itself: stored-hyperparam
# detection, primal-Newton + AD-on-θ tangent extraction, dual `Q_post`
# assembly, structural guards.

@testset "AutoDiffLikelihood IFT path (Dual hyperparams)" begin
    Random.seed!(42)
    fd_backend = AutoFiniteDiff()

    @testset "Unconstrained: scalar Dual hyperparam" begin
        # Pipeline: ForwardDiff.gradient(neg_log_posterior, θ) where
        # neg_log_posterior(θ) calls gaussian_approximation. Stored
        # hyperparams + IFT dispatch should propagate θ-Duals through the
        # posterior and back to the outer gradient.
        Random.seed!(7)
        k = 8
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        prior = model(ws; τ = 1.0, ρ = 0.3)

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
            obs_lik = obs_model(y; φ = exp(θ[1]))
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x)
        end

        θ = [0.0]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-4
    end

    @testset "Constrained: scalar Dual hyperparam (RW1 sum-to-zero)" begin
        Random.seed!(11)
        k = 8
        y = randn(k) .* 0.3
        x_eval = randn(k); x_eval .-= sum(x_eval) / k
        model = RW1Model(k)
        ws = make_workspace(model; τ = 1.0)
        prior = model(ws; τ = 1.0)

        function gauss_loglik(x; y, φ)
            return -0.5 * sum((y .- x) .^ 2) * exp(2 * φ) + length(y) * φ
        end
        obs_model = AutoDiffObservationModel(
            gauss_loglik;
            n_latent = k,
            hyperparams = (:φ,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )

        function pipeline(θ)
            obs_lik = obs_model(y; φ = θ[1])
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [log(2.0)]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-4
    end

    @testset "Multi-hyperparam + length(θ)>1 outer gradient" begin
        # Two named scalar hyperparameters carrying Duals from a length-2 θ.
        # Exercises N=2 partials in the IFT loop and the multi-hp dispatch.
        Random.seed!(17)
        k = 8
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        prior = model(ws; τ = 1.0, ρ = 0.3)

        function poisson_loglik2(x; y, φ, β)
            return sum(y .* (φ .* x .+ β) .- exp.(φ .* x .+ β))
        end
        obs_model = AutoDiffObservationModel(
            poisson_loglik2;
            n_latent = k,
            hyperparams = (:φ, :β),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )

        function pipeline(θ)
            obs_lik = obs_model(y; φ = exp(θ[1]), β = θ[2])
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x)
        end

        θ = [0.0, 0.1]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-4
    end

    @testset "Matrix-valued Dual hyperparam (shape preservation)" begin
        # Pin the contract that Matrix-shaped hyperparams round-trip through
        # the IFT path without flattening or shape errors.
        Random.seed!(13)
        k = 6
        y = randn(k) .* 0.3
        x = zeros(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        prior = model(ws; τ = 1.0, ρ = 0.3)

        function mat_loglik(x; y, W)
            s = sum(W) / size(W, 1)
            return -0.5 * sum((y .- x) .^ 2) * s
        end
        obs_model = AutoDiffObservationModel(
            mat_loglik;
            n_latent = k,
            hyperparams = (:W,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )

        function pipeline(θ)
            W = [θ[1] 0.0; 0.0 θ[1]]
            obs_lik = obs_model(y; W = W)
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x)
        end

        θ = [1.0]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test maximum(abs.(grad_fwd - grad_fd)) < 1.0e-4
    end

    @testset "Analytic AD-on-θ cross-check (Gaussian loglik)" begin
        # Validates the AD-on-θ partial extraction at machine precision
        # against a closed-form analytic derivative. Uses a Gaussian loglik
        # whose Hessian and Hessian-θ-derivative are known exactly:
        #
        #   loglik(x; y, α) = -0.5 * α * Σ(y_i - x_i)²
        #     g_i   = ∂loglik/∂x_i  = α * (y_i - x_i)
        #     ∂g_i/∂α at fixed x    = y_i - x_i
        #     H_ii  = ∂²loglik/∂x_i² = -α
        #     ∂H_ii/∂α at fixed x   = -1
        #     ∂H_ii/∂x_j            = 0 → total dH/dα equals ∂H/∂α
        #
        # Bypasses gaussian_approximation so the only error sources are AD
        # itself (exact) and floating-point rounding. The 1e-12 threshold
        # eliminates FD-oracle noise from the validation.
        function gauss_α(x; y, α)
            return -0.5 * α * sum((y .- x) .^ 2)
        end
        n_latent = 4
        obs_model = AutoDiffObservationModel(
            gauss_α;
            n_latent = n_latent,
            hyperparams = (:α,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )
        y_data = [1.0, 2.0, 3.0, 4.0]
        x_test = [0.5, 1.5, 2.5, 3.5]

        # Driver function that returns sum of grad partials so outer
        # ForwardDiff.derivative reduces to an analytic-comparable scalar.
        # Inside, it exercises exactly the AD-on-θ lift used in
        # `_autodifflik_ift_workspace` step 2.
        function grad_dα_sum(α)
            obs_lik = obs_model(y_data; α = α)
            φ = obs_lik.hyperparams.α
            if φ isa ForwardDiff.Dual
                DT = typeof(φ)
                N = ForwardDiff.npartials(DT)
                zp = ForwardDiff.Partials{N, Float64}(ntuple(_ -> 0.0, Val(N)))
                x_lifted = [DT(x_test[i], zp) for i in 1:n_latent]
                g = loggrad(x_lifted, obs_lik)
                return sum(g)
            else
                return sum(loggrad(x_test, obs_lik))
            end
        end
        α_val = 2.0
        ad_partial = ForwardDiff.derivative(grad_dα_sum, α_val)
        analytic_partial = sum(y_data .- x_test)
        @test abs(ad_partial - analytic_partial) < 1.0e-12

        # Hessian total derivative — same loglik. dH/dα at fixed x = -n
        # (since H = -α I and dH_ii/dα = -1 with no x-dependence).
        function hess_dα_sum(α)
            obs_lik = obs_model(y_data; α = α)
            φ = obs_lik.hyperparams.α
            if φ isa ForwardDiff.Dual
                DT = typeof(φ)
                N = ForwardDiff.npartials(DT)
                dx_dα = [0.3, -0.7, 0.1, 0.05]
                x_lifted = [
                    DT(x_test[i], ForwardDiff.Partials{N, Float64}((dx_dα[i],)))
                        for i in 1:n_latent
                ]
                H = loghessian(x_lifted, obs_lik)
                return sum(diag(H))
            else
                H = loghessian(x_test, obs_lik)
                return sum(diag(H))
            end
        end
        ad_hess_partial = ForwardDiff.derivative(hess_dα_sum, α_val)
        analytic_hess_partial = -float(n_latent)
        @test abs(ad_hess_partial - analytic_hess_partial) < 1.0e-12
    end

    @testset "Mismatched outer-Dual tags errors loudly" begin
        # Defensive guard: hyperparams from independent outer AD passes
        # (different Tags) would silently misread partials. The IFT
        # dispatch should error rather than dispatch.
        function loglik2(x; y, α, β)
            return sum(α .* x .+ β .* x .^ 2)
        end
        obs_model = AutoDiffObservationModel(
            loglik2;
            n_latent = 3,
            hyperparams = (:α, :β),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )
        y = [1.0, 2.0, 3.0]
        TagA = ForwardDiff.Tag{Symbol("A"), Float64}
        TagB = ForwardDiff.Tag{Symbol("B"), Float64}
        α_d = ForwardDiff.Dual{TagA, Float64, 1}(0.5, ForwardDiff.Partials{1, Float64}((1.0,)))
        β_d = ForwardDiff.Dual{TagB, Float64, 1}(0.3, ForwardDiff.Partials{1, Float64}((1.0,)))
        obs_lik = obs_model(y; α = α_d, β = β_d)
        prior = WorkspaceGMRF(zeros(3), spdiagm(0 => ones(3)))
        @test_throws ErrorException gaussian_approximation(prior, obs_lik)
    end

    @testset "_assemble_q_post_dual: sparse, dense-error, pattern-violation" begin
        # Direct unit tests of the FD-extension internals not exercised by
        # the pipeline tests above. Pipeline tests use pointwise loglik
        # (Diagonal Hessian); these cover the sparse pattern-subset method,
        # the dense-fallback explicit error, and the pattern-violation
        # error path.
        FDExt = Base.get_extension(GaussianMarkovRandomFields, :GaussianMarkovRandomFieldsForwardDiff)
        @assert FDExt !== nothing "FD extension not loaded"

        OuterTag = ForwardDiff.Tag{Symbol("test_q_assembly"), Float64}
        DualT = ForwardDiff.Dual{OuterTag, Float64, 1}
        # Trigger Tag's tagcount registration before any later comparisons.
        ForwardDiff.tagcount(OuterTag)

        Q_prior = SparseMatrixCSC(spdiagm(-1 => -ones(3), 0 => 2 * ones(4), 1 => -ones(3)))

        h_diag = [
            DualT(-0.5, ForwardDiff.Partials{1, Float64}((-1.0,))),
            DualT(-0.6, ForwardDiff.Partials{1, Float64}((-1.5,))),
            DualT(-0.7, ForwardDiff.Partials{1, Float64}((-2.0,))),
            DualT(-0.8, ForwardDiff.Partials{1, Float64}((-2.5,))),
        ]
        H_sparse = SparseMatrixCSC(spdiagm(0 => h_diag))
        Q_post_sparse = FDExt._assemble_q_post_dual(Q_prior, H_sparse, DualT, Val(1))

        @test Q_post_sparse isa SparseMatrixCSC
        @test Q_post_sparse.colptr == Q_prior.colptr
        @test Q_post_sparse.rowval == Q_prior.rowval
        for i in 1:4
            entry = Q_post_sparse[i, i]
            @test ForwardDiff.value(entry) ≈ 2.0 - ForwardDiff.value(h_diag[i])
            @test ForwardDiff.partials(entry, 1) ≈ -ForwardDiff.partials(h_diag[i], 1)
        end
        # Off-diagonals stay at Q_prior's primal value with zero partials.
        @test ForwardDiff.value(Q_post_sparse[1, 2]) ≈ -1.0
        @test ForwardDiff.partials(Q_post_sparse[1, 2], 1) == 0.0

        # Pattern violation: H has a nonzero outside Q_prior pattern.
        h_oob = copy(h_diag)
        H_violator = SparseMatrixCSC(spdiagm(0 => h_oob))
        H_violator[1, 4] = DualT(0.1, ForwardDiff.Partials{1, Float64}((0.5,)))
        @test_throws ArgumentError FDExt._assemble_q_post_dual(
            Q_prior, H_violator, DualT, Val(1)
        )

        # Dense fallback errors explicitly rather than producing a Matrix
        # that the WorkspaceGMRF constructor would reject.
        H_dense = Matrix{DualT}(H_sparse)
        @test_throws ArgumentError FDExt._assemble_q_post_dual(
            Q_prior, H_dense, DualT, Val(1)
        )
    end
end
