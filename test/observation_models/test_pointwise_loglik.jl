using GaussianMarkovRandomFields
using Test
using ReTest
using Random
using Distributions
using LinearAlgebra
using SparseArrays

@testset "Pointwise Log-Likelihood" begin
    Random.seed!(123)

    @testset "ObservationIndependence Trait System" begin
        # Test trait types exist
        @test ConditionallyIndependent <: ObservationIndependence
        @test ConditionallyDependent <: ObservationIndependence

        # Test that all current likelihoods have ConditionallyIndependent trait
        obs_model = ExponentialFamily(Normal)
        obs_lik = obs_model([1.0, 2.0, 3.0]; σ = 1.0)
        @test observation_independence(obs_lik) == ConditionallyIndependent()
    end

    @testset "ExponentialFamily: Normal" begin
        y = [1.0, 2.0, 1.5, 3.0]
        σ = 0.5
        obs_model = ExponentialFamily(Normal)
        obs_lik = obs_model(y; σ = σ)
        x = randn(length(y))

        # Test allocating version
        per_obs = pointwise_loglik(x, obs_lik)
        @test length(per_obs) == length(y)
        @test sum(per_obs) ≈ loglik(x, obs_lik)
        @test eltype(per_obs) == Float64

        # Test in-place version
        result = zeros(length(y))
        pointwise_loglik!(result, x, obs_lik)
        @test result ≈ per_obs

        # Test correctness: manually compute expected values
        expected = logpdf.(Normal.(x, σ), y)
        @test per_obs ≈ expected
    end

    @testset "ExponentialFamily: Normal with indices" begin
        y = [1.0, 2.0, 1.5]
        σ = 0.5
        indices = [1, 3, 5]
        obs_model = ExponentialFamily(Normal; indices = indices)
        obs_lik = obs_model(y; σ = σ)
        x_full = randn(10)

        per_obs = pointwise_loglik(x_full, obs_lik)
        @test length(per_obs) == length(y)  # Not length(x_full)!
        @test sum(per_obs) ≈ loglik(x_full, obs_lik)

        # Verify correct indexing
        expected = logpdf.(Normal.(x_full[indices], σ), y)
        @test per_obs ≈ expected
    end

    @testset "ExponentialFamily: Poisson" begin
        y = [1, 3, 0, 2, 5]
        obs_model = ExponentialFamily(Poisson)
        obs_lik = obs_model(y)
        x = randn(length(y))

        per_obs = pointwise_loglik(x, obs_lik)
        @test length(per_obs) == length(y)
        @test sum(per_obs) ≈ loglik(x, obs_lik)

        # Test correctness: Poisson with LogLink (default)
        λ = exp.(x)
        expected = logpdf.(Poisson.(λ), y)
        @test per_obs ≈ expected

        # Test in-place version
        result = zeros(length(y))
        pointwise_loglik!(result, x, obs_lik)
        @test result ≈ per_obs
    end

    @testset "ExponentialFamily: Poisson with offset" begin
        y = [1, 3, 0, 2]
        offset = log.([0.5, 1.0, 0.1, 2.0])  # Offset is additive on η (log scale)
        obs_model = ExponentialFamily(Poisson)
        obs_lik = obs_model(y; offset = offset)
        x = randn(length(y))

        per_obs = pointwise_loglik(x, obs_lik)
        @test length(per_obs) == length(y)
        @test sum(per_obs) ≈ loglik(x, obs_lik)

        # Test correctness: offset is additive on log scale
        λ = exp.(x .+ offset)
        expected = logpdf.(Poisson.(λ), y)
        @test per_obs ≈ expected
    end

    @testset "ExponentialFamily: Bernoulli" begin
        y = [1, 0, 1, 0, 1, 1]
        obs_model = ExponentialFamily(Bernoulli)
        obs_lik = obs_model(y)
        x = randn(length(y))

        per_obs = pointwise_loglik(x, obs_lik)
        @test length(per_obs) == length(y)
        @test sum(per_obs) ≈ loglik(x, obs_lik)

        # Test correctness: Bernoulli with LogitLink (default)
        p = 1.0 ./ (1.0 .+ exp.(-x))
        expected = logpdf.(Bernoulli.(p), y)
        @test per_obs ≈ expected

        # Test in-place version
        result = zeros(length(y))
        pointwise_loglik!(result, x, obs_lik)
        @test result ≈ per_obs
    end

    @testset "ExponentialFamily: Binomial" begin
        y = [3, 1, 4, 2]
        trials = [5, 8, 6, 10]
        bin_obs = BinomialObservations(y, trials)
        obs_model = ExponentialFamily(Binomial)
        obs_lik = obs_model(bin_obs)
        x = randn(length(y))

        per_obs = pointwise_loglik(x, obs_lik)
        @test length(per_obs) == length(y)
        @test sum(per_obs) ≈ loglik(x, obs_lik)

        # Test correctness: Binomial with LogitLink (default)
        p = 1.0 ./ (1.0 .+ exp.(-x))
        expected = logpdf.(Binomial.(trials, p), y)
        @test per_obs ≈ expected

        # Test in-place version
        result = zeros(length(y))
        pointwise_loglik!(result, x, obs_lik)
        @test result ≈ per_obs
    end

    @testset "CompositeLikelihood" begin
        # Create composite model with two components
        # Both components observe the same latent field
        y1 = [1.0, 2.0, 3.0]
        y2 = [1, 0, 1]

        model1 = ExponentialFamily(Normal)
        model2 = ExponentialFamily(Bernoulli)

        composite_model = CompositeObservationModel((model1, model2))
        y_composite = CompositeObservations((y1, y2))
        composite_lik = composite_model(y_composite; σ = 0.5)

        x = randn(3)  # Latent field (length matches observations)

        per_obs = pointwise_loglik(x, composite_lik)

        # Should concatenate per-obs log-likelihoods from both components
        @test length(per_obs) == length(y1) + length(y2)
        @test sum(per_obs) ≈ loglik(x, composite_lik)

        # Test in-place version
        result = zeros(length(y1) + length(y2))
        pointwise_loglik!(result, x, composite_lik)
        @test result ≈ per_obs

        # Verify components separately
        lik1 = model1(y1; σ = 0.5)
        lik2 = model2(y2)
        expected1 = pointwise_loglik(x, lik1)
        expected2 = pointwise_loglik(x, lik2)
        @test per_obs ≈ vcat(expected1, expected2)
    end

    @testset "LinearlyTransformedLikelihood" begin
        # Design matrix: y = A * x_full
        y = [1.0, 2.0, 1.5]
        A = [
            1.0 0.0 0.5;
            0.0 1.0 0.3;
            0.5 0.5 1.0
        ]
        A_sparse = sparse(A)

        base_model = ExponentialFamily(Normal)
        lt_model = LinearlyTransformedObservationModel(base_model, A_sparse)
        lt_lik = lt_model(y; σ = 0.5)

        x_full = randn(3)

        per_obs = pointwise_loglik(x_full, lt_lik)
        @test length(per_obs) == length(y)
        @test sum(per_obs) ≈ loglik(x_full, lt_lik)

        # Test in-place version
        result = zeros(length(y))
        pointwise_loglik!(result, x_full, lt_lik)
        @test result ≈ per_obs

        # Verify correctness: should match base likelihood evaluated at η = A * x_full
        η = A * x_full
        base_lik = base_model(y; σ = 0.5)
        expected = pointwise_loglik(η, base_lik)
        @test per_obs ≈ expected
    end

    @testset "AutoDiffLikelihood: without pointwise_loglik_func" begin
        # Test that error is raised when pointwise function not provided
        function my_loglik(x; y = [1.0, 2.0, 3.0])
            return -0.5 * sum((y .- x) .^ 2)
        end

        obs_model = AutoDiffObservationModel(my_loglik; n_latent = 3, hyperparams = (:y,))
        y_data = [1.0, 2.0, 3.0]
        obs_lik = obs_model(y_data)  # Pass y_data positionally
        x = randn(3)

        @test_throws ErrorException pointwise_loglik(x, obs_lik)
    end

    @testset "AutoDiffLikelihood: with pointwise_loglik_func" begin
        # Define both total and pointwise log-likelihood
        # These accept y as a keyword argument with a default
        function my_loglik(x; y = [1.0, 2.0, 3.0])
            return -0.5 * sum((y .- x) .^ 2)
        end

        function my_pointwise_loglik(x; y = [1.0, 2.0, 3.0])
            return -0.5 .* (y .- x) .^ 2
        end

        obs_model = AutoDiffObservationModel(
            my_loglik;
            n_latent = 3,
            hyperparams = (:y,),
            pointwise_loglik_func = my_pointwise_loglik
        )

        # Pass y_data as positional argument - gets forwarded as y= kwarg to the functions
        y_data = [1.0, 2.0, 3.0]
        obs_lik = obs_model(y_data)  # This creates closure: x -> my_loglik(x; y=y_data)
        x = randn(3)

        per_obs = pointwise_loglik(x, obs_lik)
        @test length(per_obs) == length(y_data)
        @test sum(per_obs) ≈ loglik(x, obs_lik)

        # Test in-place version
        result = zeros(length(y_data))
        pointwise_loglik!(result, x, obs_lik)
        @test result ≈ per_obs
    end

    @testset "NonlinearLeastSquaresLikelihood" begin
        # Simple nonlinear function: f(x) = x.^2
        f(x) = x .^ 2
        model = NonlinearLeastSquaresModel(f, 3)

        y = [1.0, 4.0, 9.0]
        σ = 0.5

        lik = model(y; σ = σ)
        x = [1.0, 2.0, 3.0]  # Perfect fit

        per_obs = pointwise_loglik(x, lik)
        @test length(per_obs) == length(y)
        @test sum(per_obs) ≈ loglik(x, lik)

        # Test correctness
        ŷ = f(x)
        expected = logpdf.(Normal.(ŷ, σ), y)
        @test per_obs ≈ expected

        # Test in-place version
        result = zeros(length(y))
        pointwise_loglik!(result, x, lik)
        @test result ≈ per_obs
    end

    @testset "NonlinearLeastSquares with heteroskedastic noise" begin
        f(x) = x .^ 2
        model = NonlinearLeastSquaresModel(f, 3)

        y = [1.0, 4.0, 9.0]
        σ = [0.5, 1.0, 0.3]  # Different σ for each observation

        lik = model(y; σ = σ)
        x = [1.0, 2.0, 3.0]

        per_obs = pointwise_loglik(x, lik)
        @test length(per_obs) == length(y)
        @test sum(per_obs) ≈ loglik(x, lik)

        # Test correctness
        ŷ = f(x)
        expected = logpdf.(Normal.(ŷ, σ), y)
        @test per_obs ≈ expected
    end

    @testset "Type stability" begin
        # Test that pointwise_loglik is type-stable
        y = [1.0, 2.0, 3.0]
        obs_model = ExponentialFamily(Normal)
        obs_lik = obs_model(y; σ = 1.0)
        x = randn(3)

        @inferred pointwise_loglik(x, obs_lik)
        @inferred pointwise_loglik!(zeros(3), x, obs_lik)
    end

    @testset "Non-canonical links" begin
        # Test with non-canonical link functions
        y = [1, 3, 0, 2]

        # Poisson with IdentityLink (non-canonical)
        obs_model = ExponentialFamily(Poisson, IdentityLink())
        obs_lik = obs_model(y)
        x = [0.5, 2.0, 0.1, 1.0]  # Must be positive for IdentityLink

        per_obs = pointwise_loglik(x, obs_lik)
        @test sum(per_obs) ≈ loglik(x, obs_lik)

        # Verify correctness
        λ = x  # IdentityLink: inverse is identity
        expected = logpdf.(Poisson.(λ), y)
        @test per_obs ≈ expected
    end

    @testset "Edge cases" begin
        # Test with single observation
        y = [2.0]
        obs_model = ExponentialFamily(Normal)
        obs_lik = obs_model(y; σ = 1.0)
        x = [1.5]

        per_obs = pointwise_loglik(x, obs_lik)
        @test length(per_obs) == 1
        @test per_obs[1] ≈ loglik(x, obs_lik)

        # Test with many observations
        y = randn(1000)
        obs_lik = obs_model(y; σ = 0.5)
        x = randn(1000)

        per_obs = pointwise_loglik(x, obs_lik)
        @test length(per_obs) == 1000
        @test sum(per_obs) ≈ loglik(x, obs_lik)
    end

    @testset "Interface documentation examples" begin
        # Test examples from docstrings work correctly

        # Basic usage example
        obs_model = ExponentialFamily(Poisson)
        y = [1, 3, 0, 2]
        obs_lik = obs_model(y)
        x = randn(length(y))

        per_obs = pointwise_loglik(x, obs_lik)
        total = loglik(x, obs_lik)

        @test sum(per_obs) ≈ total
        @test length(per_obs) == length(y)

        # In-place example
        result = zeros(length(y))
        pointwise_loglik!(result, x, obs_lik)
        @test sum(result) ≈ loglik(x, obs_lik)
    end
end
