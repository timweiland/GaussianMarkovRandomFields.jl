using GaussianMarkovRandomFields
using LinearAlgebra
using LinearSolve
using SparseArrays
using Distributions

@testset "Gaussian Approximation" begin

    @testset "Gaussian Likelihood - Analytical Solution" begin
        # For Gaussian likelihood, the Gaussian approximation should be exact
        n = 5
        Q_prior = Diagonal(2.0 * ones(n)) # Strong prior
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior)

        obs_model = ExponentialFamily(Normal)
        θ_named = (σ = 0.5,)
        y = [0.1, 0.2, -0.1, 0.3, -0.2]  # Small deviations

        obs_lik = obs_model(y; θ_named...)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a valid GMRF
        @test result isa GMRF
        @test length(mean(result)) == n
        @test size(precision_matrix(result)) == (n, n)

        # For Gaussian case, can verify against analytical solution
        # Posterior precision: Q_post = Q_prior + Q_obs = Q_prior + I/σ²
        σ = θ_named.σ
        Q_obs = sparse(I, n, n) / σ^2
        Q_analytical = Q_prior + Q_obs

        # Posterior mean: μ_post = Q_post^(-1) * (Q_prior * μ_prior + Q_obs * y)
        μ_analytical = Q_analytical \ (Q_prior * μ_prior + Q_obs * y)

        @test precision_matrix(result) ≈ Q_analytical atol = 1.0e-10
        @test mean(result) ≈ μ_analytical atol = 1.0e-10
    end

    @testset "Bernoulli Likelihood - Mathematical Properties" begin
        # Test with Bernoulli observation model (non-linear)
        n = 8
        Q_prior = SymTridiagonal(ones(n), fill(-0.5, n - 1))
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior, LinearSolve.LDLtFactorization())

        obs_model = ExponentialFamily(Bernoulli)
        θ_named = NamedTuple()  # No hyperparameters
        y = [1, 1, 0, 1, 0, 0, 1, 0]  # Mixed binary data

        obs_lik = obs_model(y; θ_named...)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a valid GMRF
        @test result isa GMRF
        @test length(mean(result)) == n
        @test size(precision_matrix(result)) == (n, n)

        # Mode should reflect the data pattern
        μ_result = mean(result)
        @test μ_result[1] > 0  # First observation is 1
        @test μ_result[3] < 0  # Third observation is 0

        # Precision matrix should be positive definite
        @test all(eigvals(Array(precision_matrix(result))) .> 0)
    end

    @testset "Poisson Likelihood - Mathematical Properties" begin
        # Test with Poisson observation model
        n = 6
        Q_prior = Diagonal(1.0 * ones(n))
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior)

        obs_model = ExponentialFamily(Poisson)
        θ_named = NamedTuple()  # No hyperparameters
        y = [1, 3, 0, 2, 4, 1]  # Count data

        obs_lik = obs_model(y; θ_named...)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a valid GMRF
        @test result isa GMRF
        @test length(mean(result)) == n
        @test size(precision_matrix(result)) == (n, n)

        # Mode should be reasonable for Poisson data
        μ_result = mean(result)
        @test all(isfinite.(μ_result))

        # For Poisson with log link, mode should reflect log of data pattern
        # Higher counts should correspond to higher modes
        @test μ_result[5] > μ_result[3]  # y[5]=4 > y[3]=0
        @test μ_result[2] > μ_result[3]  # y[2]=3 > y[3]=0

        # Precision matrix should be positive definite
        @test all(eigvals(Array(precision_matrix(result))) .> 0)
    end

    @testset "Prior-Posterior Consistency" begin
        # Test that approximation is reasonable relative to prior
        n = 4
        Q_prior = Diagonal(1.0 * ones(4))
        μ_prior = [1.0, -1.0, 0.5, -0.5]
        prior_gmrf = GMRF(μ_prior, Q_prior)

        obs_model = ExponentialFamily(Normal)
        θ_named = (σ = 2.0,)  # Weak likelihood
        y = μ_prior + 0.1 * randn(n)  # Data close to prior mean

        obs_lik = obs_model(y; θ_named...)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a valid GMRF
        @test result isa GMRF

        # With weak likelihood, posterior should be close to prior
        μ_result = mean(result)
        @test norm(μ_result - μ_prior) < 0.5  # Should be reasonably close

        # Posterior precision should be larger than prior precision
        @test all(diag(precision_matrix(result)) .>= diag(Q_prior))
    end

    @testset "Helper functions and edge cases" begin
        # Test neg_log_posterior function
        n = 3
        prior_gmrf = GMRF(zeros(n), sparse(I, n, n))

        # Simple normal likelihood
        obs_model = ExponentialFamily(Normal)
        y = [0.5, -0.3, 0.8]
        obs_lik = obs_model(y; σ = 1.0)

        x_test = [0.1, 0.2, -0.1]

        # Test neg_log_posterior
        neg_ll = GaussianMarkovRandomFields.neg_log_posterior(prior_gmrf, obs_lik, x_test)
        @test neg_ll isa Real
        @test isfinite(neg_ll)

        # Should be sum of negative log prior and negative log likelihood
        expected = -logpdf(prior_gmrf, x_test) - loglik(x_test, obs_lik)
        @test neg_ll ≈ expected
    end

    @testset "Indexed Gaussian approximation" begin
        # Test conjugate case with indexed observations
        n = 5
        prior_gmrf = GMRF(zeros(n), 2.0 * sparse(I, n, n))

        # Create indexed normal likelihood - observe only subset of components
        obs_indices = [1, 3, 5]  # Observe components 1, 3, 5
        y_obs = [0.5, -0.2, 0.8]
        σ = 0.3

        # Create indexed ExponentialFamily normal likelihood
        obs_model = ExponentialFamily(Normal, indices = obs_indices)
        obs_lik = obs_model(y_obs; σ = σ)

        result = gaussian_approximation(prior_gmrf, obs_lik)

        @test result isa GMRF
        @test length(mean(result)) == n

        # Verify the indexed case was used (components not observed should have prior mean)
        posterior_mean = mean(result)
        @test posterior_mean[2] ≈ 0.0 atol = 1.0e-10  # Component 2 not observed, should stay at prior mean
        @test posterior_mean[4] ≈ 0.0 atol = 1.0e-10  # Component 4 not observed, should stay at prior mean
    end

    @testset "LinearlyTransformed Gaussian approximation" begin
        # Test linearly transformed normal likelihood (conjugate case)
        n = 4
        prior_gmrf = GMRF(zeros(n), sparse(I, n, n))

        # Create design matrix A (2x4) - linear transformation
        A = sparse(
            [
                1.0 0.5 0.0 0.2;
                0.0 1.0 -0.3 0.1
            ]
        )

        # Observations
        y = [1.2, -0.5]
        σ = 0.8

        # Create base normal likelihood using ExponentialFamily
        obs_model = ExponentialFamily(Normal)
        base_lik = obs_model(y; σ = σ)

        # Create linearly transformed likelihood
        obs_lik = LinearlyTransformedLikelihood(base_lik, A)

        result = gaussian_approximation(prior_gmrf, obs_lik)

        @test result isa GMRF
        @test length(mean(result)) == n
        @test size(precision_matrix(result)) == (n, n)

        # Should be conjugate - verify it's exact analytical solution
        # Posterior precision: Q_post = Q_prior + A^T * (1/σ²) * A
        Q_prior = sparse(I, n, n)
        Q_obs = A' * (1 / σ^2) * A
        Q_expected = Q_prior + Q_obs

        @test precision_matrix(result) ≈ Q_expected atol = 1.0e-12
    end

    @testset "MetaGMRF dispatch" begin
        # Test that MetaGMRF wrapper is preserved
        n = 3
        base_gmrf = GMRF(zeros(n), sparse(I, n, n))

        # Create simple metadata subtype
        struct TestDimMetadata <: GMRFMetadata
            dim::Int
        end

        metadata = TestDimMetadata(2)
        meta_gmrf = MetaGMRF(base_gmrf, metadata)

        # Normal likelihood
        obs_model = ExponentialFamily(Normal)
        y = [0.2, -0.1, 0.4]
        obs_lik = obs_model(y; σ = 0.5)

        result = gaussian_approximation(meta_gmrf, obs_lik)

        # Should return MetaGMRF with same metadata
        @test result isa MetaGMRF
        @test result.metadata === metadata
        @test length(mean(result)) == n
    end
end
