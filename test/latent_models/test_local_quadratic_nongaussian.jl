using Test
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: local_quadratic, LocalLatentQuadratic,
    NonGaussianLatentPrior
using Distributions: logpdf
using SparseArrays
using LinearAlgebra
using Random

# Quadratic-drift state-space prior used as a `NonGaussianLatentPrior`
# fixture:
#
#   x[1] ~ N(0, 1/τ),  x[t] = a · x[t-1]² + N(0, 1/τ)  for t ≥ 2.
#
# log p(x|τ,a) = ½ n log(τ) - ½ n log(2π)
#              - ½τ ( x[1]² + Σ_{t=2}^n (x[t] - a·x[t-1]²)² ).
#
# The Hessian depends on x via the (x[t] - a·x[t-1]²)² coupling, so the
# prior has no canonical materialised GMRF — exactly the case the
# `NonGaussianLatentPrior` interface is for.
struct QuadraticDriftModel <: NonGaussianLatentPrior
    n::Int
end

Base.length(m::QuadraticDriftModel) = m.n
GaussianMarkovRandomFields.hyperparameters(::QuadraticDriftModel) = (τ = Real, a = Real)
GaussianMarkovRandomFields.model_name(::QuadraticDriftModel) = :quadratic_drift
GaussianMarkovRandomFields.constraints(::QuadraticDriftModel; kwargs...) = nothing

# Exact joint log-density of the prior at x.
function _qd_logp(x, τ::Real, a::Real)
    n = length(x)
    s = x[1]^2
    @inbounds for t in 2:n
        s += (x[t] - a * x[t - 1]^2)^2
    end
    return 0.5 * n * log(τ) - 0.5 * n * log(2π) - 0.5 * τ * s
end

# Exact gradient ∇ log p.
function _qd_grad(x, τ::Real, a::Real)
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

# Exact `Q = -∇² log p` as sparse tridiagonal. Each ψ_t = (x[t] - a·x[t-1]²)²
# enters log p with coefficient `-½τ`, so its contribution to Q at any
# (i, j) is `+½τ · ∂²ψ_t/∂x[i]∂x[j]`.
function _qd_neg_hessian(x, τ::Real, a::Real)
    n = length(x)
    rows = Int[]; cols = Int[]; vals = Float64[]
    # ψ_1 = x[1]² ⇒ Q[1,1] = +½τ · 2 = τ
    push!(rows, 1); push!(cols, 1); push!(vals, τ)
    @inbounds for t in 2:n
        r = x[t] - a * x[t - 1]^2
        # ∂²ψ_t/∂x[t]² = 2 ⇒ Q[t,t] += τ
        push!(rows, t); push!(cols, t); push!(vals, τ)
        # ∂²ψ_t/∂x[t-1]² = -4a·r + 8a²·x[t-1]²
        d2_xtm1 = -4 * a * r + 8 * a^2 * x[t - 1]^2
        push!(rows, t - 1); push!(cols, t - 1); push!(vals, 0.5 * τ * d2_xtm1)
        # ∂²ψ_t/∂x[t-1]∂x[t] = -4a·x[t-1]
        d_off = 0.5 * τ * (-4 * a * x[t - 1])
        push!(rows, t - 1); push!(cols, t); push!(vals, d_off)
        push!(rows, t); push!(cols, t - 1); push!(vals, d_off)
    end
    return sparse(rows, cols, vals, n, n)
end

function GaussianMarkovRandomFields.local_quadratic(
        m::QuadraticDriftModel, x_ref::AbstractVector;
        τ::Real, a::Real,
    )
    Q = _qd_neg_hessian(x_ref, τ, a)
    g = _qd_grad(x_ref, τ, a)
    h = g + Q * x_ref
    logp_ref = _qd_logp(x_ref, τ, a)
    return LocalLatentQuadratic(Q, h, logp_ref, x_ref)
end

@testset "local_quadratic — non-Gaussian prior validation" begin

    @testset "Hessian/gradient analytic vs FiniteDiff" begin
        # Sanity-check the analytic Q, h before using them as ground truth.
        Random.seed!(1)
        n = 5
        τ = 1.0; a = 0.3
        x = randn(n) .* 0.4
        ε = 1.0e-6

        # Gradient
        g_an = _qd_grad(x, τ, a)
        g_fd = similar(x)
        for i in eachindex(x)
            xp = copy(x); xp[i] += ε
            xm = copy(x); xm[i] -= ε
            g_fd[i] = (_qd_logp(xp, τ, a) - _qd_logp(xm, τ, a)) / (2ε)
        end
        @test g_an ≈ g_fd atol = 1.0e-7

        # Hessian: -Q is the Hessian of log p.
        Q_an = _qd_neg_hessian(x, τ, a)
        H_an = -Matrix(Q_an)
        H_fd = zeros(n, n)
        for i in 1:n, j in 1:n
            xpp = copy(x); xpp[i] += ε; xpp[j] += ε
            xpm = copy(x); xpm[i] += ε; xpm[j] -= ε
            xmp = copy(x); xmp[i] -= ε; xmp[j] += ε
            xmm = copy(x); xmm[i] -= ε; xmm[j] -= ε
            H_fd[i, j] = (
                _qd_logp(xpp, τ, a) - _qd_logp(xpm, τ, a) -
                    _qd_logp(xmp, τ, a) + _qd_logp(xmm, τ, a)
            ) / (4 * ε^2)
        end
        @test H_an ≈ H_fd atol = 1.0e-3
    end

    @testset "Posterior mode matches numerical optimum" begin
        # Reference mode: damped Newton on the *exact* negative log-joint
        # with FD-checked gradient/Hessian.
        Random.seed!(3)
        n = 4
        τ = 2.0; a = 0.6
        σ = 0.5
        x_true = zeros(n)
        for t in 2:n
            x_true[t] = a * x_true[t - 1]^2 + (1 / sqrt(τ)) * randn()
        end
        y = x_true .+ σ .* randn(n)

        model = QuadraticDriftModel(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)

        post = gaussian_approximation(model, obs_lik; τ = τ, a = a)

        x_gt = zeros(n)
        for _ in 1:200
            g = -_qd_grad(x_gt, τ, a) .- (y .- x_gt) ./ σ^2
            Q = _qd_neg_hessian(x_gt, τ, a) + (1 / σ^2) * sparse(1.0I, n, n)
            δ = Q \ g
            x_gt -= δ
            norm(δ, Inf) < 1.0e-10 && break
        end

        @test mean(post) ≈ x_gt atol = 1.0e-6
        Q_post_expected = _qd_neg_hessian(x_gt, τ, a) + (1 / σ^2) * sparse(1.0I, n, n)
        @test Matrix(precision_matrix(post)) ≈ Matrix(Q_post_expected) atol = 1.0e-4
    end

    @testset "Laplace marginal: non-Gaussian-prior bias is bounded" begin
        # For non-Gaussian priors the Laplace approximation has bias
        # proportional to higher derivatives of log p at the mode. With
        # a concentrated posterior (high τ) the bias is small; we
        # verify it's within a few percent of brute-force integration.
        Random.seed!(9)
        n = 3
        τ = 10.0; a = 0.4
        σ = 0.4
        y = [0.3, 0.4, 0.5]

        model = QuadraticDriftModel(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)

        post = gaussian_approximation(model, obs_lik; τ = τ, a = a)
        logml_laplace = marginal_loglikelihood(model, obs_lik, post; τ = τ, a = a)

        x_star = mean(post)
        Q_post = Matrix(precision_matrix(post))
        std_dev = sqrt.(diag(inv(Q_post)))
        m_axis = 80
        ranges = [range(x_star[i] - 8 * std_dev[i], x_star[i] + 8 * std_dev[i], length = m_axis) for i in 1:n]
        dV = prod(step.(ranges))
        S = 0.0
        for i in 1:m_axis, j in 1:m_axis, k in 1:m_axis
            x_pt = [ranges[1][i], ranges[2][j], ranges[3][k]]
            logp = _qd_logp(x_pt, τ, a) + sum(logpdf.(Normal.(x_pt, σ), y))
            S += exp(logp) * dV
        end
        logml_brute = log(S)
        # Bounded by Laplace's higher-order error (~σ_post² × κ where κ
        # is local curvature / nonlinearity). Empirical: ≤ 1% here.
        @test abs(logml_laplace - logml_brute) / abs(logml_brute) < 1.0e-2
    end

    @testset "Convergence from a difficult initial point" begin
        # Far-from-mode start with backtracking line search must still
        # converge to the same mode the prior-mean start reaches, with
        # an SPD posterior precision.
        Random.seed!(31)
        n = 4
        τ = 1.0; a = 0.7
        σ = 0.5
        y = randn(n) .* 0.3 .+ 0.5
        model = QuadraticDriftModel(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)

        post_ref = gaussian_approximation(model, obs_lik; τ = τ, a = a)
        x_hard = fill(-2.0, n)
        post_hard = gaussian_approximation(
            model, obs_lik; τ = τ, a = a, x0 = x_hard, max_iter = 200
        )

        @test mean(post_hard) ≈ mean(post_ref) atol = 1.0e-4
        @test isposdef(Symmetric(Matrix(precision_matrix(post_hard))))
    end

    @testset "Iterated linearisation through GMRFWorkspace" begin
        # The workspace path must agree with the cache path. The workspace
        # is seeded with a tridiagonal pattern matching `local_quadratic`'s
        # pattern at non-zero x.
        Random.seed!(43)
        n = 4
        τ = 2.0; a = 0.5
        σ = 0.5
        y = [0.4, 0.7, 1.0, 1.1]
        model = QuadraticDriftModel(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)

        post_cache = gaussian_approximation(model, obs_lik; τ = τ, a = a)

        Q_pattern = spdiagm(
            -1 => fill(-0.01, n - 1), 0 => fill(float(τ), n), 1 => fill(-0.01, n - 1)
        )
        ws = GaussianMarkovRandomFields.GMRFWorkspace(Q_pattern)
        post_ws = gaussian_approximation(model, obs_lik; τ = τ, a = a, ws = ws)

        @test mean(post_ws) ≈ mean(post_cache) atol = 1.0e-6
        @test Matrix(precision_matrix(post_ws)) ≈ Matrix(precision_matrix(post_cache)) atol = 1.0e-4
        @test post_ws isa GaussianMarkovRandomFields.WorkspaceGMRF
    end

    @testset "Bias regression: iterated vs linearise-once" begin
        # Linearise-once-at-zero (Q at x=0 collapses to τ·I for the
        # quadratic drift) disagrees with iterated linearisation at the
        # mode whenever the data pulls x* away from zero.
        Random.seed!(5)
        n = 5
        τ = 1.5; a = 0.8       # strong nonlinearity
        σ = 0.4
        y = [0.6, 1.1, 1.4, 1.6, 1.7]

        model = QuadraticDriftModel(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)

        post_iter = gaussian_approximation(model, obs_lik; τ = τ, a = a)

        lq_at_zero = local_quadratic(model, zeros(n); τ = τ, a = a)
        fixed_gmrf = GMRF(zeros(n), lq_at_zero.Q)
        post_fixed = gaussian_approximation(fixed_gmrf, obs_lik)

        diff = norm(mean(post_iter) - mean(post_fixed))
        @test diff > 1.0e-3

        # Iterated mode satisfies the *exact* joint optimality condition;
        # fixed mode does not.
        x_iter = mean(post_iter)
        x_fixed = mean(post_fixed)
        ∇joint_iter = _qd_grad(x_iter, τ, a) .+ (y .- x_iter) ./ σ^2
        ∇joint_fixed = _qd_grad(x_fixed, τ, a) .+ (y .- x_fixed) ./ σ^2
        @test norm(∇joint_iter, Inf) < 1.0e-6
        @test norm(∇joint_fixed, Inf) > 1.0e-3
    end
end
