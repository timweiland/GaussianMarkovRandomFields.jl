using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random

@testset "ConstrainedGMRF" begin
    rng = Random.MersenneTwister(12345)

    # Create a simple unconstrained GMRF for testing
    μ = [1.0, 2.0, 3.0, 4.0]
    Q = spdiagm(0 => [1.0, 2.0, 3.0, 4.0])
    base_gmrf = GMRF(μ, Q)

    @testset "Sum-to-zero constraint" begin
        # Simple sum-to-zero constraint: sum(x) = 0
        A = ones(1, 4)  # constraint matrix
        e = [0.0]       # constraint vector

        constrained = ConstrainedGMRF(base_gmrf, A, e)

        @testset "Basic properties" begin
            @test length(constrained) == 4
            @test isa(constrained, AbstractGMRF)
        end

        @testset "Mean constraint satisfaction" begin
            μ_constrained = mean(constrained)
            @test sum(μ_constrained) ≈ 0.0 atol = 1.0e-10
            @test length(μ_constrained) == 4
        end

        @testset "Sampling constraint satisfaction" begin
            # Test multiple samples to ensure they all satisfy the constraint
            for i in 1:10
                sample = rand(rng, constrained)
                @test sum(sample) ≈ 0.0 atol = 1.0e-10
                @test length(sample) == 4
            end
        end

        @testset "Variance properties" begin
            var_constrained = var(constrained)
            var_unconstrained = var(base_gmrf)

            @test length(var_constrained) == 4
            @test all(var_constrained .≥ 0)  # variances should be non-negative
            @test mean(var_constrained) < mean(var_unconstrained)  # should reduce variance on average
        end

        @testset "Empirical mean & variance validation" begin
            # Get theoretical mean and variances
            theoretical_mean = mean(constrained)
            theoretical_var = var(constrained)

            # Generate many samples to estimate empirical statistics
            rng_empirical = Random.MersenneTwister(12345)
            samps = [rand(rng_empirical, constrained) for i in 1:50000]
            empirical_mean = mean(samps)
            empirical_var = var(samps)

            # Test that empirical and theoretical statistics match
            @test empirical_mean ≈ theoretical_mean rtol = 0.05
            @test empirical_var ≈ theoretical_var rtol = 0.05
        end

        @testset "Precision map" begin
            Q_constrained = precision_map(constrained)
            @test size(Q_constrained) == (4, 4)
            # For now, we return the base precision map (TODO: implement proper constrained precision)
            # This is a placeholder implementation
            @test Q_constrained == precision_map(base_gmrf)
        end
    end

    @testset "Multiple constraints" begin
        # Two constraints: sum(x) = 0 and x[1] - x[2] = 1
        A = [1.0 1.0 1.0 1.0; 1.0 -1.0 0.0 0.0]
        e = [0.0, 1.0]

        constrained = ConstrainedGMRF(base_gmrf, A, e)

        @testset "Mean constraint satisfaction" begin
            μ_constrained = mean(constrained)
            @test A * μ_constrained ≈ e atol = 1.0e-10
        end

        @testset "Sampling constraint satisfaction" begin
            for i in 1:10
                sample = rand(rng, constrained)
                @test A * sample ≈ e atol = 1.0e-10
            end
        end
    end

    @testset "Single element constraint" begin
        # Fix first element to a specific value: x[1] = 5.0
        A = [1.0 0.0 0.0 0.0]'  # transpose to make it a column vector, then transpose again
        A = reshape([1.0, 0.0, 0.0, 0.0], 1, 4)
        e = [5.0]

        constrained = ConstrainedGMRF(base_gmrf, A, e)

        @testset "Mean constraint satisfaction" begin
            μ_constrained = mean(constrained)
            @test μ_constrained[1] ≈ 5.0 atol = 1.0e-10
        end

        @testset "Sampling constraint satisfaction" begin
            for i in 1:5
                sample = rand(rng, constrained)
                @test sample[1] ≈ 5.0 atol = 1.0e-10
            end
        end

        @testset "Variance reduction" begin
            var_constrained = var(constrained)
            @test var_constrained[1] ≈ 0.0 atol = 1.0e-10  # constrained element should have zero variance
            @test all(var_constrained[2:4] .> 0)  # unconstrained elements should have positive variance
        end
    end

    @testset "Edge cases" begin
        @testset "Zero constraint vector" begin
            A = [1.0 1.0 0.0 0.0]
            A = reshape(A, 1, 4)
            e = [0.0]

            constrained = ConstrainedGMRF(base_gmrf, A, e)
            μ_constrained = mean(constrained)
            @test A * μ_constrained ≈ e atol = 1.0e-10
        end

        @testset "Different base GMRF sizes" begin
            # Test with different sized GMRF
            μ_small = [1.0, 2.0]
            Q_small = spdiagm(0 => [1.0, 1.0])
            base_small = GMRF(μ_small, Q_small)

            A_small = reshape([1.0, 1.0], 1, 2)
            e_small = [0.0]

            constrained_small = ConstrainedGMRF(base_small, A_small, e_small)
            @test length(constrained_small) == 2
            @test sum(mean(constrained_small)) ≈ 0.0 atol = 1.0e-10
        end
    end

    @testset "logpdf" begin
        # Sum-to-zero constraint: sum(x) = 0
        A = ones(1, 4)
        e = [0.0]
        constrained = ConstrainedGMRF(base_gmrf, A, e)

        @testset "Valid points (satisfy constraint)" begin
            # Sample from the constrained distribution (should satisfy constraint)
            rng_test = Random.MersenneTwister(42)
            sample = rand(rng_test, constrained)

            # Verify constraint is satisfied
            @test A * sample ≈ e atol = 1.0e-10

            # logpdf should be finite for valid points
            lpdf = logpdf(constrained, sample)
            @test isfinite(lpdf)

            # Test at the constrained mean
            lpdf_mean = logpdf(constrained, mean(constrained))
            @test isfinite(lpdf_mean)
        end

        @testset "Invalid points (violate constraint)" begin
            # Create a point that doesn't satisfy the constraint
            invalid_point = [1.0, 2.0, 3.0, 4.0]  # sum = 10, not 0
            @test !(A * invalid_point ≈ e)

            # logpdf should be -Inf for points that violate constraints
            lpdf_invalid = logpdf(constrained, invalid_point)
            @test lpdf_invalid == -Inf
        end

        @testset "Relative probabilities" begin
            # Points closer to the mean should have higher logpdf
            rng_test = Random.MersenneTwister(123)
            samples = [rand(rng_test, constrained) for _ in 1:5]

            logpdfs = [logpdf(constrained, s) for s in samples]

            # All should be finite
            @test all(isfinite.(logpdfs))

            # logpdf at mean should be highest (or among the highest)
            lpdf_at_mean = logpdf(constrained, mean(constrained))
            @test lpdf_at_mean >= minimum(logpdfs)
        end
    end
end
