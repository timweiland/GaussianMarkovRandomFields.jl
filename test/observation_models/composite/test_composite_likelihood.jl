using Distributions: Normal, Poisson
using LinearAlgebra: Diagonal
using SparseArrays: SparseMatrixCSC

@testset "CompositeLikelihood evaluation" begin
    @testset "Basic loglik summation" begin
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)
        composite_model = CompositeObservationModel((gaussian_model, poisson_model))

        y1 = [1.0, 2.0]  # 2 Gaussian observations
        y2 = PoissonObservations([3, 4])      # 2 Poisson observations
        y_composite = CompositeObservations((y1, y2))

        composite_lik = composite_model(y_composite; σ = 1.0)

        # Both components target the full 2D latent field (since both y's are 2D)
        x = randn(2)
        ll_composite = loglik(x, composite_lik)

        # Should equal manual summation - both components see full x
        gaussian_lik = gaussian_model(y1; σ = 1.0)
        poisson_lik = poisson_model(y2)
        ll_manual = loglik(x, gaussian_lik) + loglik(x, poisson_lik)

        @test ll_composite ≈ ll_manual
    end

    @testset "Gradient summation" begin
        gaussian_model = ExponentialFamily(Normal)
        composite_model = CompositeObservationModel((gaussian_model, gaussian_model))

        y1 = [1.0, 2.0]  # 2 observations
        y2 = [3.0, 4.0]  # 2 observations
        y_composite = CompositeObservations((y1, y2))

        composite_lik = composite_model(y_composite; σ = 1.0)

        # Both components see the 2D latent field
        x = randn(2)
        grad_composite = loggrad(x, composite_lik)

        # Should equal sum of gradients - both components see full x
        lik1 = gaussian_model(y1; σ = 1.0)
        lik2 = gaussian_model(y2; σ = 1.0)
        grad_manual = loggrad(x, lik1) + loggrad(x, lik2)

        @test grad_composite ≈ grad_manual
    end

    @testset "Hessian summation" begin
        gaussian_model = ExponentialFamily(Normal)
        composite_model = CompositeObservationModel((gaussian_model, gaussian_model))

        y1 = [1.0]  # 1 observation -> 1D latent field
        y2 = [2.0]  # 1 observation -> 1D latent field
        y_composite = CompositeObservations((y1, y2))

        composite_lik = composite_model(y_composite; σ = 1.0)

        # Both components see the 1D latent field
        x = randn(1)
        hess_composite = loghessian(x, composite_lik)

        # Should equal sum of Hessians - both components see full x
        lik1 = gaussian_model(y1; σ = 1.0)
        lik2 = gaussian_model(y2; σ = 1.0)

        hess_manual = loghessian(x, lik1) + loghessian(x, lik2)

        @test hess_composite ≈ hess_manual
        @test size(hess_composite) == (1, 1)
    end

    @testset "Mixed likelihood types" begin
        # Test that different likelihood types can be combined
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)
        composite_model = CompositeObservationModel((gaussian_model, poisson_model))

        y_composite = CompositeObservations(([1.0, 2.0], PoissonObservations([3, 4])))  # Both 2D
        composite_lik = composite_model(y_composite; σ = 1.5)

        x = randn(2)  # 2D latent field to match observations

        # Should be able to evaluate with mixed types
        ll = loglik(x, composite_lik)
        @test ll isa Float64

        grad = loggrad(x, composite_lik)
        @test length(grad) == 2

        hess = loghessian(x, composite_lik)
        @test size(hess) == (2, 2)
    end

    @testset "Hessian sum across mismatched component types" begin
        # AutoDiffObservationModel with default sparse Hessian backend returns
        # SparseMatrixCSC; ExponentialFamily(Normal) returns Diagonal. The
        # accumulator must promote correctly regardless of component order.
        function band_loglik(x; y)
            ll = 0.0
            for i in eachindex(y)
                ll += -0.5 * (x[i] - y[i])^2
                if i > 1
                    ll += -0.1 * x[i - 1] * x[i]
                end
            end
            return ll
        end
        autodiff_model = AutoDiffObservationModel(band_loglik; n_latent = 4)
        gaussian_model = ExponentialFamily(Normal)

        y_ad = [0.5, 1.5, 2.5, 3.5]
        y_g = [1.0, 2.0, 3.0, 4.0]

        ad_lik = autodiff_model(y_ad)
        g_lik = gaussian_model(y_g; σ = 1.0)
        @test loghessian(zeros(4), ad_lik) isa SparseMatrixCSC
        @test loghessian(zeros(4), g_lik) isa Diagonal

        x = randn(4)
        hess_manual = loghessian(x, ad_lik) + loghessian(x, g_lik)

        # Sparse first, then Diagonal — route σ only to the gaussian component.
        composite_lik_a = CompositeObservationModel(
            (autodiff_model, gaussian_model),
            (NamedTuple(), (σ = :σ,)),
        )(CompositeObservations((y_ad, y_g)); σ = 1.0)
        @test loghessian(x, composite_lik_a) ≈ hess_manual

        # Diagonal first, then Sparse
        composite_lik_b = CompositeObservationModel(
            (gaussian_model, autodiff_model),
            ((σ = :σ,), NamedTuple()),
        )(CompositeObservations((y_g, y_ad)); σ = 1.0)
        @test loghessian(x, composite_lik_b) ≈ hess_manual
    end

    @testset "Type stability" begin
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)
        composite_model = CompositeObservationModel((gaussian_model, poisson_model))

        y_composite = CompositeObservations(([1.0], PoissonObservations([2])))  # Both 1D -> 1D latent field
        composite_lik = composite_model(y_composite; σ = 1.0)

        x = randn(1)  # 1D latent field

        # All evaluation methods should be type stable
        @inferred loglik(x, composite_lik)
        @inferred loggrad(x, composite_lik)
        @inferred loghessian(x, composite_lik)
    end
end
