using Test
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Distributions

@testset "Gaussian Approximation - ChordalGMRF" begin

    @testset "Gaussian Likelihood - Analytical Solution" begin
        # For Gaussian likelihood, the Gaussian approximation should be exact
        n = 5
        Q_prior = spdiagm(0 => 2.0 * ones(n))
        μ_prior = zeros(n)
        prior_gmrf = ChordalGMRF(μ_prior, Q_prior)

        obs_model = ExponentialFamily(Normal)
        y = [0.1, 0.2, -0.1, 0.3, -0.2]

        obs_lik = obs_model(y; σ = 0.5)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a ChordalGMRF
        @test result isa ChordalGMRF
        @test length(mean(result)) == n

        # For Gaussian case, verify against analytical solution
        σ = 0.5
        Q_obs = sparse(I, n, n) / σ^2
        Q_analytical = Q_prior + Q_obs
        μ_analytical = Q_analytical \ (Q_prior * μ_prior + Q_obs * y)

        @test precision_matrix(result) ≈ Q_analytical atol = 1e-8
        @test mean(result) ≈ μ_analytical atol = 1e-8
    end

    @testset "Bernoulli Likelihood - Mathematical Properties" begin
        # Test with Bernoulli observation model (non-linear)
        n = 8
        Q_prior = spdiagm(0 => ones(n), 1 => fill(-0.3, n-1), -1 => fill(-0.3, n-1))
        μ_prior = zeros(n)
        prior_gmrf = ChordalGMRF(μ_prior, Q_prior)

        obs_model = ExponentialFamily(Bernoulli)
        y = [1, 1, 0, 1, 0, 0, 1, 0]

        obs_lik = obs_model(y)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a ChordalGMRF
        @test result isa ChordalGMRF
        @test length(mean(result)) == n

        # Mode should reflect the data pattern
        μ_result = mean(result)
        @test μ_result[1] > 0  # First observation is 1
        @test μ_result[3] < 0  # Third observation is 0
    end

    @testset "Poisson Likelihood - Mathematical Properties" begin
        # Test with Poisson observation model
        n = 6
        Q_prior = spdiagm(0 => ones(n))
        μ_prior = zeros(n)
        prior_gmrf = ChordalGMRF(μ_prior, Q_prior)

        obs_model = ExponentialFamily(Poisson)
        y = PoissonObservations([1, 3, 0, 2, 4, 1])

        obs_lik = obs_model(y)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a ChordalGMRF
        @test result isa ChordalGMRF
        @test length(mean(result)) == n

        # Mode should be reasonable for Poisson data
        μ_result = mean(result)
        @test all(isfinite.(μ_result))

        # Higher counts should correspond to higher modes
        @test μ_result[5] > μ_result[3]  # y[5]=4 > y[3]=0
        @test μ_result[2] > μ_result[3]  # y[2]=3 > y[3]=0
    end

    @testset "Consistency with GMRF" begin
        # Results should match between GMRF and ChordalGMRF
        n = 5
        Q_prior = spdiagm(0 => 2.0 * ones(n), 1 => fill(-0.5, n-1), -1 => fill(-0.5, n-1))
        μ_prior = zeros(n)

        gmrf_prior = GMRF(μ_prior, Q_prior)
        chordal_prior = ChordalGMRF(μ_prior, Q_prior)

        obs_model = ExponentialFamily(Poisson)
        y = PoissonObservations([2, 5, 1, 3, 4])
        obs_lik = obs_model(y)

        result_gmrf = gaussian_approximation(gmrf_prior, obs_lik)
        result_chordal = gaussian_approximation(chordal_prior, obs_lik)

        @test mean(result_gmrf) ≈ mean(result_chordal) atol = 1e-6
        @test precision_matrix(result_gmrf) ≈ precision_matrix(result_chordal) atol = 1e-6
    end

    @testset "Sparse precision - tridiagonal" begin
        # Test with tridiagonal precision (common in GMRFs)
        n = 10
        Q_prior = spdiagm(0 => 2.0 * ones(n), 1 => -ones(n-1), -1 => -ones(n-1))
        μ_prior = zeros(n)
        prior_gmrf = ChordalGMRF(μ_prior, Q_prior)

        obs_model = ExponentialFamily(Normal)
        y = randn(n)
        obs_lik = obs_model(y; σ = 0.5)

        result = gaussian_approximation(prior_gmrf, obs_lik)

        @test result isa ChordalGMRF
        @test length(mean(result)) == n
        @test all(isfinite.(mean(result)))
    end

    @testset "Warm-start with x0" begin
        n = 6
        Q_prior = spdiagm(0 => ones(n))
        prior_gmrf = ChordalGMRF(zeros(n), Q_prior)

        obs_model = ExponentialFamily(Poisson)
        y = PoissonObservations([5, 12, 2, 8, 15, 3])
        obs_lik = obs_model(y)

        # Cold-start result
        result_cold = gaussian_approximation(prior_gmrf, obs_lik)
        x_star = mean(result_cold)

        # Warm-start from converged mode
        result_warm = gaussian_approximation(prior_gmrf, obs_lik; x0 = x_star)
        @test mean(result_warm) ≈ x_star atol = 1e-4
    end

    @testset "Adaptive stepsize - extreme Poisson" begin
        # Test case where adaptive stepsize is needed
        n = 3
        Q_prior = spdiagm(0 => 0.01 * ones(n))  # Weak prior
        prior_gmrf = ChordalGMRF(zeros(n), Q_prior)

        obs_model = ExponentialFamily(Poisson)
        y = PoissonObservations([100, 500, 50])
        obs_lik = obs_model(y)

        result = gaussian_approximation(prior_gmrf, obs_lik)

        @test result isa ChordalGMRF
        μ_result = mean(result)
        @test all(isfinite.(μ_result))

        # Each component should be close to log of its count
        for i in 1:n
            @test abs(μ_result[i] - log(y.counts[i])) < 1.5
        end
    end

    @testset "Non-convergence path" begin
        # Force non-convergence with max_iter = 1
        n = 5
        Q_prior = spdiagm(0 => ones(n))
        prior_gmrf = ChordalGMRF(zeros(n), Q_prior)

        obs_model = ExponentialFamily(Poisson)
        y = PoissonObservations([10, 15, 8, 12, 20])
        obs_lik = obs_model(y)

        result = gaussian_approximation(prior_gmrf, obs_lik; max_iter = 1)

        # Should still return a ChordalGMRF
        @test result isa ChordalGMRF
        @test all(isfinite.(mean(result)))
    end

    @testset "Verbose output" begin
        n = 4
        Q_prior = spdiagm(0 => ones(n))
        prior_gmrf = ChordalGMRF(zeros(n), Q_prior)

        obs_model = ExponentialFamily(Poisson)
        y = PoissonObservations([2, 3, 1, 4])
        obs_lik = obs_model(y)

        # Should not error with verbose=true
        result = gaussian_approximation(prior_gmrf, obs_lik; verbose = true)
        @test result isa ChordalGMRF
    end

end
