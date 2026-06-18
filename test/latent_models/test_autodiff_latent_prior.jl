using Test
using GaussianMarkovRandomFields
import DifferentiationInterface as DI
using ForwardDiff
# Loading these activates the SparseADLikelihoods extension, so the AD prior's
# default Hessian backend is sparse (Tracer detection + greedy colouring).
using SparseConnectivityTracer, SparseMatrixColorings
using SparseArrays
using LinearAlgebra
using Random
using Distributions: Normal, logpdf

# Self-contained quadratic-drift joint log-prior with analytic derivatives as ground
# truth. x[1] ~ N(0, 1/τ), x[t] = a·x[t-1]² + N(0, 1/τ). The Hessian depends on x
# (genuinely non-Gaussian); the off-diagonals are structural but vanish at x = 0.
function _qd_logp(x, τ, a)
    n = length(x)
    s = x[1]^2
    @inbounds for t in 2:n
        s += (x[t] - a * x[t - 1]^2)^2
    end
    return 0.5 * n * log(τ) - 0.5 * n * log(2π) - 0.5 * τ * s
end

function _qd_grad(x, τ, a)
    n = length(x)
    g = zeros(n)
    g[1] = -τ * x[1]
    @inbounds for t in 2:n
        r = x[t] - a * x[t - 1]^2
        g[t - 1] += τ * r * 2 * a * x[t - 1]
        g[t] += -τ * r
    end
    return g
end

function _qd_neg_hessian(x, τ, a)
    n = length(x)
    rows = Int[]; cols = Int[]; vals = Float64[]
    push!(rows, 1); push!(cols, 1); push!(vals, τ)
    @inbounds for t in 2:n
        r = x[t] - a * x[t - 1]^2
        push!(rows, t); push!(cols, t); push!(vals, τ)
        d2 = -4 * a * r + 8 * a^2 * x[t - 1]^2
        push!(rows, t - 1); push!(cols, t - 1); push!(vals, 0.5 * τ * d2)
        doff = 0.5 * τ * (-4 * a * x[t - 1])
        push!(rows, t - 1); push!(cols, t); push!(vals, doff)
        push!(rows, t); push!(cols, t - 1); push!(vals, doff)
    end
    return sparse(rows, cols, vals, n, n)
end

# The AD prior wraps the same density as a keyword-θ function.
_qd_logp_kw(x; τ, a) = _qd_logp(x, τ, a)

@testset "AutoDiffLatentPrior" begin

    @testset "local_quadratic matches analytic (sparse AD)" begin
        Random.seed!(1)
        n = 5; τ = 1.5; a = 0.4
        prior = AutoDiffLatentPrior(_qd_logp_kw; n = n, hyperparams = (:τ, :a))
        for _ in 1:5
            x = randn(n) .* 0.5
            lq = local_quadratic(prior, x; τ = τ, a = a)
            @test lq.Q isa SparseMatrixCSC
            @test Matrix(lq.Q) ≈ Matrix(_qd_neg_hessian(x, τ, a)) atol = 1.0e-8
            @test lq.h ≈ _qd_grad(x, τ, a) .+ _qd_neg_hessian(x, τ, a) * x atol = 1.0e-8
            @test lq.logp_ref ≈ _qd_logp(x, τ, a) atol = 1.0e-10
        end
    end

    @testset "prior_logdensity is the cheap primal and agrees with local_quadratic" begin
        n = 4; τ = 2.0; a = 0.6
        prior = AutoDiffLatentPrior(_qd_logp_kw; n = n, hyperparams = (:τ, :a))
        x = randn(n) .* 0.5
        @test prior_logdensity(prior, x; τ = τ, a = a) == _qd_logp(x, τ, a)
        @test prior_logdensity(prior, x; τ = τ, a = a) ≈
            local_quadratic(prior, x; τ = τ, a = a).logp_ref
    end

    @testset "AbstractLatentPrior interface" begin
        prior = AutoDiffLatentPrior(_qd_logp_kw; n = 7, hyperparams = (:τ, :a), name = :qd)
        @test length(prior) == 7
        @test GaussianMarkovRandomFields.model_name(prior) == :qd
        @test GaussianMarkovRandomFields.constraints(prior; τ = 1.0, a = 0.0) === nothing
        @test GaussianMarkovRandomFields.hyperparameters(prior) == (τ = Real, a = Real)
    end

    @testset "gaussian_approximation matches an independent Newton on the joint" begin
        Random.seed!(3)
        n = 4; τ = 2.0; a = 0.6; σ = 0.5
        x_true = zeros(n)
        for t in 2:n
            x_true[t] = a * x_true[t - 1]^2 + (1 / sqrt(τ)) * randn()
        end
        y = x_true .+ σ .* randn(n)
        prior = AutoDiffLatentPrior(_qd_logp_kw; n = n, hyperparams = (:τ, :a))
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)
        post = gaussian_approximation(prior, obs_lik; τ = τ, a = a)

        # Independent damped Newton on the exact joint negative log-posterior.
        x_gt = zeros(n)
        for _ in 1:200
            g = -_qd_grad(x_gt, τ, a) .- (y .- x_gt) ./ σ^2
            Q = _qd_neg_hessian(x_gt, τ, a) + (1 / σ^2) * sparse(1.0I, n, n)
            δ = Q \ g
            x_gt -= δ
            norm(δ, Inf) < 1.0e-10 && break
        end
        @test mean(post) ≈ x_gt atol = 1.0e-6
        Q_expected = _qd_neg_hessian(x_gt, τ, a) + (1 / σ^2) * sparse(1.0I, n, n)
        @test Matrix(precision_matrix(post)) ≈ Matrix(Q_expected) atol = 1.0e-4
    end

    @testset "structured: AD latent prior + exact Normal likelihood, marginal Laplace" begin
        # The AD prior composes with a closed-form likelihood; the Laplace marginal
        # matches brute-force integration up to Laplace's higher-order bias.
        Random.seed!(9)
        n = 3; τ = 10.0; a = 0.4; σ = 0.4
        y = [0.3, 0.4, 0.5]
        prior = AutoDiffLatentPrior(_qd_logp_kw; n = n, hyperparams = (:τ, :a))
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)
        post = gaussian_approximation(prior, obs_lik; τ = τ, a = a)
        @test isposdef(Symmetric(Matrix(precision_matrix(post))))

        logml = marginal_loglikelihood(prior, obs_lik, post; τ = τ, a = a)
        x_star = mean(post)
        Q_post = Matrix(precision_matrix(post))
        sd = sqrt.(diag(inv(Q_post)))
        m_axis = 80
        ranges = [range(x_star[i] - 8 * sd[i], x_star[i] + 8 * sd[i], length = m_axis) for i in 1:n]
        dV = prod(step.(ranges))
        S = 0.0
        for i in 1:m_axis, j in 1:m_axis, k in 1:m_axis
            xp = [ranges[1][i], ranges[2][j], ranges[3][k]]
            logp = _qd_logp(xp, τ, a) + sum(logpdf.(Normal.(xp, σ), y))
            S += exp(logp) * dV
        end
        @test abs(logml - log(S)) / abs(log(S)) < 1.0e-2
    end

    @testset "make_workspace + workspace GA matches the cache path" begin
        # The structural (value-agnostic) Hessian sparsity lets make_workspace seed
        # from x_ref = zeros; the workspace path must then agree with the cache path
        # (and the prior's per-iterate Q pattern must stay fixed, else the workspace
        # update would error).
        Random.seed!(43)
        n = 4; τ = 2.0; a = 0.5; σ = 0.5
        y = [0.4, 0.7, 1.0, 1.1]
        prior = AutoDiffLatentPrior(_qd_logp_kw; n = n, hyperparams = (:τ, :a))
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)

        post_cache = gaussian_approximation(prior, obs_lik; τ = τ, a = a)
        ws = make_workspace(prior; τ = τ, a = a)
        post_ws = gaussian_approximation(prior, obs_lik; τ = τ, a = a, ws = ws)

        @test post_ws isa GaussianMarkovRandomFields.WorkspaceGMRF
        @test mean(post_ws) ≈ mean(post_cache) atol = 1.0e-6
        @test Matrix(precision_matrix(post_ws)) ≈ Matrix(precision_matrix(post_cache)) atol = 1.0e-4
    end

    @testset "ZeroLikelihood evaluates to nothing" begin
        x = randn(5)
        @test loglik(x, ZeroLikelihood()) == 0.0
        @test loggrad(x, ZeroLikelihood()) == zeros(5)
        @test loghessian(x, ZeroLikelihood()) == Diagonal(zeros(5))
    end

    @testset "monolithic joint via ZeroLikelihood == structured (prior + likelihood)" begin
        # Design (a): a monolithic TMB joint is an AutoDiffLatentPrior carrying ALL the
        # energy + ZeroLikelihood. When the joint factorises as prior + likelihood, it
        # must give the same posterior and Laplace marginal as the structured form.
        Random.seed!(7)
        n = 4; τ = 2.0; a = 0.6; σ = 0.5
        y = [0.4, 0.7, 1.0, 1.1]

        # Structured: AD prior for the drift + an exact, closed-form Normal likelihood.
        drift_prior = AutoDiffLatentPrior(_qd_logp_kw; n = n, hyperparams = (:τ, :a))
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)
        post_structured = gaussian_approximation(drift_prior, obs_lik; τ = τ, a = a)

        # Monolithic: drift log-prior + Normal log-likelihood folded into one joint.
        # The Normal term is written as plain arithmetic (not `Normal(μ, σ)`) so the
        # sparse-Hessian tracer can trace through it — see the AutoDiffLatentPrior note.
        # `joint` captures the data array `y`; the default Enzyme backend can't prove
        # such a capture read-only, so use ForwardDiff here (see the docstring note).
        joint(x; τ, a) = _qd_logp(x, τ, a) -
            0.5 * sum(((y .- x) ./ σ) .^ 2) - n * (log(σ) + 0.5 * log(2π))
        joint_prior = AutoDiffLatentPrior(
            joint; n = n, hyperparams = (:τ, :a), grad_backend = DI.AutoForwardDiff()
        )
        post_joint = gaussian_approximation(joint_prior, ZeroLikelihood(); τ = τ, a = a)

        @test mean(post_joint) ≈ mean(post_structured) atol = 1.0e-6
        @test Matrix(precision_matrix(post_joint)) ≈ Matrix(precision_matrix(post_structured)) atol = 1.0e-4

        ml_structured = marginal_loglikelihood(drift_prior, obs_lik, post_structured; τ = τ, a = a)
        ml_joint = marginal_loglikelihood(joint_prior, ZeroLikelihood(), post_joint; τ = τ, a = a)
        @test ml_joint ≈ ml_structured atol = 1.0e-6
    end

    @testset "θ-gradients of the marginal likelihood via IFT match finite differences" begin
        # Exact ForwardDiff θ-gradients of the Laplace marginal, through the IFT path
        # (primal Newton + analytic θ-tangent), validated against central differences.
        Random.seed!(11)
        n = 4; σ = 0.5
        y = [0.4, 0.7, 1.0, 1.1]
        prior = AutoDiffLatentPrior(_qd_logp_kw; n = n, hyperparams = (:τ, :a))
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)

        function ml(θvec)
            τ, a = θvec[1], θvec[2]
            post = gaussian_approximation(prior, obs_lik; τ = τ, a = a)
            return marginal_loglikelihood(prior, obs_lik, post; τ = τ, a = a)
        end

        θ0 = [2.0, 0.5]
        g_ad = ForwardDiff.gradient(ml, θ0)

        h = 1.0e-6
        g_fd = similar(θ0)
        for i in 1:2
            θp = copy(θ0); θp[i] += h
            θm = copy(θ0); θm[i] -= h
            g_fd[i] = (ml(θp) - ml(θm)) / (2h)
        end
        @test g_ad ≈ g_fd rtol = 1.0e-4

        # The IFT also yields a Dual posterior whose mean carries the mode sensitivity
        # dx*/dθ; check that Jacobian against finite differences (uses a proper AD tag).
        mode(θvec) = mean(gaussian_approximation(prior, obs_lik; τ = θvec[1], a = θvec[2]))
        J_ad = ForwardDiff.jacobian(mode, θ0)
        J_fd = similar(J_ad)
        for i in 1:2
            θp = copy(θ0); θp[i] += h
            θm = copy(θ0); θm[i] -= h
            J_fd[:, i] .= (mode(θp) .- mode(θm)) ./ (2h)
        end
        @test J_ad ≈ J_fd rtol = 1.0e-4
    end

    @testset "θ-gradients via IFT reuse a GMRFWorkspace and match the no-ws path (#174)" begin
        # The IFT hyperparameter-gradient path accepts a seeded workspace: the primal Newton
        # reuses ws's symbolic factorisation and its converged Q_post factor backs the IFT
        # solves, so one workspace serves the whole θ-grid instead of re-running the symbolic
        # analysis every gradient evaluation. The result must match the fresh-cache no-ws path.
        Random.seed!(11)
        n = 4; σ = 0.5
        y = [0.4, 0.7, 1.0, 1.1]
        prior = AutoDiffLatentPrior(_qd_logp_kw; n = n, hyperparams = (:τ, :a))
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)
        ws = make_workspace(prior; τ = 2.0, a = 0.5)

        ml_ws(θ) = marginal_loglikelihood(
            prior, obs_lik,
            gaussian_approximation(prior, obs_lik; τ = θ[1], a = θ[2], ws = ws);
            τ = θ[1], a = θ[2],
        )
        ml_nows(θ) = marginal_loglikelihood(
            prior, obs_lik,
            gaussian_approximation(prior, obs_lik; τ = θ[1], a = θ[2]);
            τ = θ[1], a = θ[2],
        )

        θ0 = [2.0, 0.5]
        g_ws = ForwardDiff.gradient(ml_ws, θ0)      # previously threw: ws unsupported on the IFT path
        @test g_ws ≈ ForwardDiff.gradient(ml_nows, θ0) rtol = 1.0e-6

        # ...and still matches central differences of the (workspace) marginal likelihood.
        h = 1.0e-6
        g_fd = similar(θ0)
        for i in 1:2
            θp = copy(θ0); θp[i] += h
            θm = copy(θ0); θm[i] -= h
            g_fd[i] = (ml_ws(θp) - ml_ws(θm)) / (2h)
        end
        @test g_ws ≈ g_fd rtol = 1.0e-4

        # One seeded workspace is reusable across the θ-grid: a gradient at a second θ also
        # matches no-ws (sequential reuse is valid — the structural pattern is θ-independent).
        θ1 = [3.0, 0.3]
        @test ForwardDiff.gradient(ml_ws, θ1) ≈ ForwardDiff.gradient(ml_nows, θ1) rtol = 1.0e-6
    end
end
