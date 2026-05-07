using Distributions: Normal, Poisson

@testset "CompositeObservationModel routing" begin
    @testset "Backward compat: single-arg constructor, disjoint kwargs" begin
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)
        composite_model = CompositeObservationModel((gaussian_model, poisson_model))

        @test composite_model.routes === (nothing, nothing)

        y_composite = CompositeObservations(([1.0, 2.0], PoissonObservations([3, 4])))
        composite_lik = composite_model(y_composite; σ = 1.5)
        @test composite_lik isa CompositeLikelihood

        x = randn(2)
        ll_composite = loglik(x, composite_lik)
        ll_manual = loglik(x, gaussian_model([1.0, 2.0]; σ = 1.5)) +
            loglik(x, poisson_model(PoissonObservations([3, 4])))
        @test ll_composite ≈ ll_manual
    end

    @testset "Backward compat: shared kwarg, two Normals see same σ" begin
        gaussian_model = ExponentialFamily(Normal)
        composite_model = CompositeObservationModel((gaussian_model, gaussian_model))

        y_composite = CompositeObservations(([1.0, 2.0], [3.0, 4.0]))
        composite_lik = composite_model(y_composite; σ = 2.0)

        x = randn(2)
        ll_composite = loglik(x, composite_lik)
        ll_manual = loglik(x, gaussian_model([1.0, 2.0]; σ = 2.0)) +
            loglik(x, gaussian_model([3.0, 4.0]; σ = 2.0))
        @test ll_composite ≈ ll_manual
    end

    @testset "Routes resolve σ collision between two Normals" begin
        m_phys = ExponentialFamily(Normal)
        m_data = ExponentialFamily(Normal)
        composite_model = CompositeObservationModel(
            (m_phys, m_data),
            ((σ = :σ_phys,), (σ = :σ_data,)),
        )

        y_composite = CompositeObservations(([0.5, 1.5], [10.0, 20.0]))
        composite_lik = composite_model(y_composite; σ_phys = 0.1, σ_data = 5.0)

        x = randn(2)
        ll_composite = loglik(x, composite_lik)
        ll_manual = loglik(x, m_phys([0.5, 1.5]; σ = 0.1)) +
            loglik(x, m_data([10.0, 20.0]; σ = 5.0))
        @test ll_composite ≈ ll_manual

        grad_composite = loggrad(x, composite_lik)
        grad_manual = loggrad(x, m_phys([0.5, 1.5]; σ = 0.1)) +
            loggrad(x, m_data([10.0, 20.0]; σ = 5.0))
        @test grad_composite ≈ grad_manual

        hess_composite = loghessian(x, composite_lik)
        hess_manual = loghessian(x, m_phys([0.5, 1.5]; σ = 0.1)) +
            loghessian(x, m_data([10.0, 20.0]; σ = 5.0))
        @test hess_composite ≈ hess_manual

        # Distinct σ values must produce distinct results
        flipped_lik = composite_model(y_composite; σ_phys = 5.0, σ_data = 0.1)
        @test loglik(x, flipped_lik) ≉ ll_composite
    end

    @testset "Mixed routing: one component routed, other passthrough" begin
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)
        composite_model = CompositeObservationModel(
            (gaussian_model, poisson_model),
            ((σ = :σ_obs,), nothing),
        )

        y_composite = CompositeObservations(([1.0, 2.0], PoissonObservations([3, 4])))
        composite_lik = composite_model(y_composite; σ_obs = 0.7)

        x = randn(2)
        ll_composite = loglik(x, composite_lik)
        ll_manual = loglik(x, gaussian_model([1.0, 2.0]; σ = 0.7)) +
            loglik(x, poisson_model(PoissonObservations([3, 4])))
        @test ll_composite ≈ ll_manual
    end

    @testset "Validation" begin
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)

        @test_throws ArgumentError CompositeObservationModel(
            (gaussian_model, poisson_model),
            ((σ = :σ_only,),),  # length mismatch
        )

        composite_model = CompositeObservationModel(
            (gaussian_model, poisson_model),
            ((σ = :σ_obs,), nothing),
        )
        y_composite = CompositeObservations(([1.0], PoissonObservations([2])))
        @test_throws ErrorException composite_model(y_composite; σ_other = 1.0)
    end

    @testset "Type stability with routes" begin
        m_phys = ExponentialFamily(Normal)
        m_data = ExponentialFamily(Normal)
        composite_model = CompositeObservationModel(
            (m_phys, m_data),
            ((σ = :σ_phys,), (σ = :σ_data,)),
        )

        y_composite = CompositeObservations(([1.0], [2.0]))
        composite_lik = composite_model(y_composite; σ_phys = 0.5, σ_data = 1.5)

        x = randn(1)
        @inferred loglik(x, composite_lik)
        @inferred loggrad(x, composite_lik)
        @inferred loghessian(x, composite_lik)
    end

    @testset "Show method skips trivial routes, displays explicit ones" begin
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)

        bc_model = CompositeObservationModel((gaussian_model, poisson_model))
        bc_str = sprint(show, bc_model)
        @test !occursin("σ_phys", bc_str)
        @test !occursin("route", lowercase(bc_str))

        routed_model = CompositeObservationModel(
            (gaussian_model, gaussian_model),
            ((σ = :σ_phys,), (σ = :σ_data,)),
        )
        routed_str = sprint(show, routed_model)
        @test occursin("σ_phys", routed_str)
        @test occursin("σ_data", routed_str)
    end
end
