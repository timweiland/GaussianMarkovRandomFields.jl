using Test
using GaussianMarkovRandomFields
using Distributions: Poisson, logpdf
using SparseArrays
using LinearAlgebra
using Random

# For Gaussian latent priors, `gaussian_approximation(model, obs_lik; θ...)`
# must match `gaussian_approximation(model(; θ...), obs_lik)` to machine
# precision: both go through fixed-Q Newton on the same materialised GMRF.

const _LQ_RTOL = 1.0e-12

function _assert_equivalent_posteriors(p1, p2; rtol = _LQ_RTOL)
    @test mean(p1) ≈ mean(p2) rtol = rtol
    return @test precision_matrix(p1) ≈ precision_matrix(p2) rtol = rtol
end

@testset "local_quadratic — Gaussian latent prior regression" begin

    @testset "AR1Model + Poisson likelihood" begin
        Random.seed!(7)
        n = 12
        y = rand(Poisson(2), n)
        obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))

        model = AR1Model(n)
        θ = (τ = 1.5, ρ = 0.4)

        post_gmrf = gaussian_approximation(model(; θ...), obs_lik)
        post_model = gaussian_approximation(model, obs_lik; θ...)

        _assert_equivalent_posteriors(post_model, post_gmrf)
    end

    @testset "RW1Model (intrinsic) + Poisson likelihood" begin
        Random.seed!(11)
        n = 10
        y = rand(Poisson(2), n)
        obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))

        model = RW1Model(n)
        θ = (τ = 1.0,)

        post_gmrf = gaussian_approximation(model(; θ...), obs_lik)
        post_model = gaussian_approximation(model, obs_lik; θ...)

        _assert_equivalent_posteriors(post_model, post_gmrf)
    end

    @testset "RW2Model (intrinsic) + Poisson likelihood" begin
        Random.seed!(13)
        n = 12
        y = rand(Poisson(2), n)
        obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))

        model = RW2Model(n)
        θ = (τ = 1.0,)

        post_gmrf = gaussian_approximation(model(; θ...), obs_lik)
        post_model = gaussian_approximation(model, obs_lik; θ...)

        _assert_equivalent_posteriors(post_model, post_gmrf)
    end

    @testset "IIDModel + Poisson likelihood" begin
        Random.seed!(17)
        n = 8
        y = rand(Poisson(2), n)
        obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))

        model = IIDModel(n)
        θ = (τ = 1.0,)

        post_gmrf = gaussian_approximation(model(; θ...), obs_lik)
        post_model = gaussian_approximation(model, obs_lik; θ...)

        _assert_equivalent_posteriors(post_model, post_gmrf)
    end

    @testset "Hyperparams via splat or NamedTuple are equivalent" begin
        # Both calling conventions should give identical posteriors.
        n = 6
        y = rand(Poisson(2), n)
        obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))
        model = AR1Model(n)

        post_splat = gaussian_approximation(model, obs_lik; τ = 1.5, ρ = 0.4)
        post_tuple = gaussian_approximation(model, obs_lik; θ = (τ = 1.5, ρ = 0.4))
        _assert_equivalent_posteriors(post_splat, post_tuple)
    end

    @testset "Laplace marginal log-likelihood: Gaussian prior + Gaussian lik (exact)" begin
        # Gaussian prior + conjugate Gaussian likelihood ⇒ Laplace is exact.
        # `marginal_loglikelihood` must match brute-force integration up to
        # the quadrature grid error.
        Random.seed!(19)
        n = 3
        σ = 0.5
        y = randn(n) .* 0.3
        model = IIDModel(n)
        θ = (τ = 4.0,)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)
        post = gaussian_approximation(model, obs_lik; θ...)
        logml_lap = marginal_loglikelihood(model, obs_lik, post; θ...)

        x_star = mean(post)
        Q_post = Matrix(precision_matrix(post))
        std_dev = sqrt.(diag(inv(Q_post)))
        m_axis = 80
        ranges = [range(x_star[i] - 8 * std_dev[i], x_star[i] + 8 * std_dev[i], length = m_axis) for i in 1:n]
        dV = prod(step.(ranges))
        S = 0.0
        for i in 1:m_axis, j in 1:m_axis, k in 1:m_axis
            x_pt = [ranges[1][i], ranges[2][j], ranges[3][k]]
            logp = logpdf(model(; θ...), x_pt) + sum(logpdf.(Normal.(x_pt, σ), y))
            S += exp(logp) * dV
        end
        logml_brute = log(S)
        @test abs(logml_lap - logml_brute) / abs(logml_brute) < 1.0e-3
    end

    @testset "Laplace marginal log-likelihood: constrained Gaussian prior" begin
        # RW1 sum-to-zero prior + Gaussian likelihood. Laplace is exact
        # (Gaussian + Gaussian). The brute-force reference integrates over
        # the (n-1)-dim constraint manifold by parametrising x via a basis
        # of the constraint's null space; this gives an exact Lebesgue
        # marginal that the Laplace formula must match.
        Random.seed!(23)
        n = 3
        σ = 0.5
        y = randn(n) .* 0.4
        model = RW1Model(n)
        θ = (τ = 2.0,)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ)
        post = gaussian_approximation(model, obs_lik; θ...)
        logml_lap = marginal_loglikelihood(model, obs_lik, post; θ...)

        # Brute-force: parametrise x ∈ {Ax = 0} via a null-space basis
        # `N` (orthonormal columns spanning the n-1 dim subspace), so
        # x = N · z with z ∈ R^{n-1}. Integrate the joint over z.
        prior_gmrf = model(; θ...)
        x_star = mean(post)
        # For RW1's sum-to-zero constraint A = [1 1 ... 1] / sqrt(n),
        # any orthonormal basis of its null space works.
        A = ones(1, n) ./ sqrt(n)
        N = nullspace(A)               # n × (n-1)
        Q_post = Matrix(precision_matrix(post))
        std_z = sqrt.(diag(inv(N' * Q_post * N)))
        m_axis = 80
        z_ranges = [range(-8 * std_z[i], 8 * std_z[i], length = m_axis) for i in 1:(n - 1)]
        dV = prod(step.(z_ranges))
        S = 0.0
        for i in 1:m_axis, j in 1:m_axis
            z = [z_ranges[1][i], z_ranges[2][j]]
            x_pt = x_star + N * z
            logp = logpdf(prior_gmrf, x_pt) + sum(logpdf.(Normal.(x_pt, σ), y))
            S += exp(logp) * dV
        end
        logml_brute = log(S)
        @test abs(logml_lap - logml_brute) / abs(logml_brute) < 1.0e-3
    end
end
