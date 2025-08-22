using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Distributions

@testset "Constrained Gaussian Approximation" begin
    # Setup: small constrained GMRF with sum-to-zero constraint
    n = 4
    Q_prior = spdiagm(0 => ones(n))
    base_gmrf = GMRF(zeros(n), Q_prior)
    A = ones(1, n)
    e = [0.0]
    prior_constrained = ConstrainedGMRF(base_gmrf, A, e)

    @testset "Normal observations" begin
        obs_model = ExponentialFamily(Normal)
        y = [0.5, -0.2, 0.1, -0.4]
        obs_lik = obs_model(y; σ = 1.0)

        result = gaussian_approximation(prior_constrained, obs_lik)

        @test result isa ConstrainedGMRF
        @test sum(mean(result)) ≈ 0.0 atol = 1.0e-10
        @test norm(mean(result) - mean(prior_constrained)) > 0.01  # Should move from prior
    end

    @testset "Bernoulli observations" begin
        obs_model = ExponentialFamily(Bernoulli)
        y = [1, 0, 1, 0]
        obs_lik = obs_model(y)

        result = gaussian_approximation(prior_constrained, obs_lik; max_iter = 20)

        @test result isa ConstrainedGMRF
        @test sum(mean(result)) ≈ 0.0 atol = 1.0e-10
        @test all(isfinite.(mean(result)))
    end

    @testset "Multiple constraints" begin
        # Two constraints: sum = 0 and x[1] = x[2]
        A_multi = [1.0 1.0 1.0 1.0; 1.0 -1.0 0.0 0.0]
        e_multi = [0.0, 0.0]
        prior_multi = ConstrainedGMRF(base_gmrf, A_multi, e_multi)

        obs_model = ExponentialFamily(Normal)
        y = [0.1, 0.2, -0.1, -0.2]
        obs_lik = obs_model(y; σ = 0.5)

        result = gaussian_approximation(prior_multi, obs_lik)

        @test result isa ConstrainedGMRF
        @test A_multi * mean(result) ≈ e_multi atol = 1.0e-10
    end

    @testset "Conjugate cases" begin
        # Test linear_condition dispatch (use sparse matrices to avoid type issues)
        A_obs = sparse([1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0])  # Observe pairs (sparse)
        y_linear = [0.2, -0.1]
        Q_ϵ = 4.0  # High precision observations

        result_linear = linear_condition(prior_constrained; A = A_obs, Q_ϵ = Q_ϵ, y = y_linear)
        @test result_linear isa ConstrainedGMRF
        @test sum(mean(result_linear)) ≈ 0.0 atol = 1.0e-10

        # Test NormalLikelihood conjugate dispatch (should use linear_condition internally)
        obs_model_normal = ExponentialFamily(Normal)
        obs_lik_normal = obs_model_normal([0.1, 0.0, -0.1, 0.0]; σ = 1.0)

        result_normal = gaussian_approximation(prior_constrained, obs_lik_normal)
        @test result_normal isa ConstrainedGMRF
        @test sum(mean(result_normal)) ≈ 0.0 atol = 1.0e-10
    end
end
