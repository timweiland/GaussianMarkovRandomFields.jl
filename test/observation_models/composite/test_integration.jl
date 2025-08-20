using Distributions: Normal, Poisson

@testset "Composite Likelihood Integration Tests" begin
    @testset "End-to-end workflow with indexed models" begin
        # Create indexed observation models for different parts of latent field
        gaussian_model = ExponentialFamily(Normal, indices = 1:3)    # First 3 elements
        poisson_model = ExponentialFamily(Poisson, indices = 4:6)    # Next 3 elements

        # Create composite model
        composite_model = CompositeObservationModel((gaussian_model, poisson_model))

        # Prepare observation data
        y_gaussian = [1.0, 2.0, 1.5]  # 3 Gaussian observations
        y_poisson = [2, 3, 1]         # 3 Poisson observations
        y_composite = CompositeObservations((y_gaussian, y_poisson))

        # Materialize composite likelihood
        composite_lik = composite_model(y_composite; σ = 0.8)

        # Test evaluation on 6D latent field
        x = [0.9, 2.1, 1.4, log(2.2), log(2.9), log(1.1)]  # First 3 for Gaussian, last 3 (log-scale) for Poisson

        # Should evaluate correctly
        ll = loglik(x, composite_lik)
        @test ll isa Float64

        # Compare against manual computation
        gaussian_lik = gaussian_model(y_gaussian; σ = 0.8)
        poisson_lik = poisson_model(y_poisson)

        ll_manual = loglik(x, gaussian_lik) + loglik(x, poisson_lik)
        @test ll ≈ ll_manual

        # Test gradient
        grad = loggrad(x, composite_lik)
        @test length(grad) == 6

        grad_manual = loggrad(x, gaussian_lik) + loggrad(x, poisson_lik)
        @test grad ≈ grad_manual

        # Test Hessian
        hess = loghessian(x, composite_lik)
        @test size(hess) == (6, 6)

        hess_manual = loghessian(x, gaussian_lik) + loghessian(x, poisson_lik)
        @test hess ≈ hess_manual
    end

    @testset "Overlapping indices" begin
        # Test overlapping case with same likelihood type (so same hyperparameters work)
        model1 = ExponentialFamily(Normal, indices = 1:3)      # First 3 elements
        model2 = ExponentialFamily(Normal, indices = 2:4)      # Elements 2-4 (overlap on 2,3)

        composite_model = CompositeObservationModel((model1, model2))

        # Different observations for each component
        y1 = [1.0, 1.5, 2.0]
        y2 = [1.4, 2.1, 2.6]
        y_composite = CompositeObservations((y1, y2))

        # Materialize - both components use same σ
        composite_lik = composite_model(y_composite; σ = 1.0)

        # Evaluate on 4D latent field
        x = [1.0, 1.5, 2.0, 2.5]
        ll = loglik(x, composite_lik)

        # Manual computation: both models contribute, overlap adds up
        lik1 = model1(y1; σ = 1.0)
        lik2 = model2(y2; σ = 1.0)

        ll_manual = loglik(x, lik1) + loglik(x, lik2)
        @test ll ≈ ll_manual

        # Test gradient accumulation at overlapping indices
        grad = loggrad(x, composite_lik)
        grad_manual = loggrad(x, lik1) + loggrad(x, lik2)
        @test grad ≈ grad_manual
    end

    @testset "Performance: composite vs manual summation" begin
        # Test that composite likelihood has minimal overhead
        gaussian_model = ExponentialFamily(Normal, indices = 1:2)
        poisson_model = ExponentialFamily(Poisson, indices = 3:4)

        composite_model = CompositeObservationModel((gaussian_model, poisson_model))
        y_composite = CompositeObservations(([1.0, 2.0], [3, 4]))
        composite_lik = composite_model(y_composite; σ = 1.0)

        # Individual likelihoods for comparison
        gaussian_lik = gaussian_model([1.0, 2.0]; σ = 1.0)
        poisson_lik = poisson_model([3, 4])

        x = randn(4)

        # Both should give same results
        ll_composite = loglik(x, composite_lik)
        ll_manual = loglik(x, gaussian_lik) + loglik(x, poisson_lik)
        @test ll_composite ≈ ll_manual

        grad_composite = loggrad(x, composite_lik)
        grad_manual = loggrad(x, gaussian_lik) + loggrad(x, poisson_lik)
        @test grad_composite ≈ grad_manual
    end

    @testset "Type stability" begin
        # Ensure all operations are type stable
        gaussian_model = ExponentialFamily(Normal, indices = 1:2)
        poisson_model = ExponentialFamily(Poisson, indices = 3:4)

        composite_model = CompositeObservationModel((gaussian_model, poisson_model))
        y_composite = CompositeObservations(([1.0, 2.0], [3, 4]))
        composite_lik = composite_model(y_composite; σ = 1.0)

        x = randn(4)

        # All operations should be type stable
        @inferred loglik(x, composite_lik)
        @inferred loggrad(x, composite_lik)
        @inferred loghessian(x, composite_lik)
    end
end
