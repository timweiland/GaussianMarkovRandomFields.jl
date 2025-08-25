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

    @testset "Indexed Normal observations" begin
        # Test ConstrainedGMRF with indexed Normal observations (conjugate case)
        obs_model = ExponentialFamily(Normal, indices = [1, 3])  # Observe only components 1 and 3
        y = [0.3, -0.2]
        obs_lik = obs_model(y; σ = 0.5)

        result = gaussian_approximation(prior_constrained, obs_lik)

        @test result isa ConstrainedGMRF
        @test sum(mean(result)) ≈ 0.0 atol = 1.0e-10  # Constraint still satisfied
        # Components 2 and 4 should be less affected since they weren't observed
        posterior_mean = mean(result)
        @test abs(posterior_mean[2]) < abs(posterior_mean[1])  # Component 2 less affected
        @test abs(posterior_mean[4]) < abs(posterior_mean[3])  # Component 4 less affected
    end

    @testset "LinearlyTransformed Normal observations" begin
        # Test ConstrainedGMRF with linearly transformed Normal observations (conjugate case)
        A_obs = sparse([1.0 0.5 0.0 0.2; 0.0 1.0 -0.3 0.1])  # 2x4 design matrix
        y = [0.8, -0.3]

        obs_model = ExponentialFamily(Normal)
        base_lik = obs_model(y; σ = 0.4)
        obs_lik = LinearlyTransformedLikelihood(base_lik, A_obs)

        result = gaussian_approximation(prior_constrained, obs_lik)

        @test result isa ConstrainedGMRF
        @test sum(mean(result)) ≈ 0.0 atol = 1.0e-10  # Constraint still satisfied
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
