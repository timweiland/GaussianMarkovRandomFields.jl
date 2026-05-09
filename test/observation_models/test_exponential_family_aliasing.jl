using Distributions
using ForwardDiff
using LinearAlgebra
using Random

@testset "ExponentialFamily kwarg aliasing" begin
    @testset "Backward compat: no aliases" begin
        ef = ExponentialFamily(Normal)
        @test ef.kwarg_aliases === nothing
        @test hyperparameters(ef) == (:σ,)

        y = [0.5, 1.0, -0.3]
        lik = ef(y; σ = 0.7)
        x = randn(3)
        @test isfinite(loglik(x, lik))
    end

    @testset "Normal: σ → outer name" begin
        ef_alias = ExponentialFamily(Normal; σ = :σ_phys)
        ef_plain = ExponentialFamily(Normal)

        y = [0.5, 1.0, -0.3]
        x = randn(3)

        lik_alias = ef_alias(y; σ_phys = 0.7)
        lik_plain = ef_plain(y; σ = 0.7)

        @test loglik(x, lik_alias) ≈ loglik(x, lik_plain)
        @test loggrad(x, lik_alias) ≈ loggrad(x, lik_plain)
        @test Matrix(loghessian(x, lik_alias)) ≈ Matrix(loghessian(x, lik_plain))
    end

    @testset "Normal alias preserves indices" begin
        ef = ExponentialFamily(Normal; σ = :σ_obs, indices = 2:4)
        @test ef.indices == 2:4
        @test ef.kwarg_aliases === (σ = :σ_obs,)

        y = [1.0, 2.0, 3.0]
        lik = ef(y; σ_obs = 1.5)
        @test lik.indices == 2:4
        @test lik.σ == 1.5
    end

    @testset "Gamma: phi → outer name" begin
        ef_alias = ExponentialFamily(Gamma; phi = :phi_obs)
        ef_plain = ExponentialFamily(Gamma)

        y = [1.5, 0.3, 4.2]
        x = log.([1.0, 2.0, 3.0])  # LogLink: μ = exp(x) > 0

        lik_alias = ef_alias(y; phi_obs = 2.0)
        lik_plain = ef_plain(y; phi = 2.0)

        @test loglik(x, lik_alias) ≈ loglik(x, lik_plain)
        @test loggrad(x, lik_alias) ≈ loggrad(x, lik_plain)
    end

    @testset "TDist: both σ and ν renamed" begin
        ef_alias = ExponentialFamily(TDist; σ = :σ_t, ν = :ν_t)
        ef_plain = ExponentialFamily(TDist)

        y = [1.5, -0.3, 4.2]
        x = randn(3)

        lik_alias = ef_alias(y; σ_t = 1.0, ν_t = 5.0)
        lik_plain = ef_plain(y; σ = 1.0, ν = 5.0)

        @test loglik(x, lik_alias) ≈ loglik(x, lik_plain)
        @test loggrad(x, lik_alias) ≈ loggrad(x, lik_plain)
        @test Matrix(loghessian(x, lik_alias)) ≈ Matrix(loghessian(x, lik_plain))
    end

    @testset "TDist: only one of two renamed" begin
        ef_alias = ExponentialFamily(TDist; σ = :σ_obs)
        ef_plain = ExponentialFamily(TDist)

        y = [1.5, -0.3, 4.2]
        x = randn(3)

        lik_alias = ef_alias(y; σ_obs = 1.0, ν = 5.0)
        lik_plain = ef_plain(y; σ = 1.0, ν = 5.0)

        @test loglik(x, lik_alias) ≈ loglik(x, lik_plain)
    end

    @testset "NegativeBinomial: r → outer name" begin
        ef_alias = ExponentialFamily(NegativeBinomial; r = :r_obs)
        ef_plain = ExponentialFamily(NegativeBinomial)

        y = NegativeBinomialObservations([2, 5, 1])
        x = randn(3)

        lik_alias = ef_alias(y; r_obs = 3.0)
        lik_plain = ef_plain(y; r = 3.0)

        @test loglik(x, lik_alias) ≈ loglik(x, lik_plain)
        @test loggrad(x, lik_alias) ≈ loggrad(x, lik_plain)
    end

    @testset "conditional_distribution honours aliases" begin
        ef_alias = ExponentialFamily(Normal; σ = :σ_phys)
        ef_plain = ExponentialFamily(Normal)

        x = [0.5, -0.2, 1.1]
        d_alias = conditional_distribution(ef_alias, x; σ_phys = 0.8)
        d_plain = conditional_distribution(ef_plain, x; σ = 0.8)

        @test mean(d_alias) ≈ mean(d_plain)
        @test var(d_alias) ≈ var(d_plain)
    end

    @testset "Validation: unknown alias key" begin
        @test_throws ArgumentError ExponentialFamily(Normal; foo = :bar)
        @test_throws ArgumentError ExponentialFamily(Gamma; σ = :σ_obs)
    end

    @testset "Validation: alias value must be Symbol" begin
        @test_throws ArgumentError ExponentialFamily(Normal; σ = "σ_phys")
        @test_throws ArgumentError ExponentialFamily(Normal; σ = 1.0)
    end

    @testset "Type stability with aliases" begin
        ef = ExponentialFamily(Normal; σ = :σ_phys)
        y = [0.5, 1.0]
        lik = ef(y; σ_phys = 0.7)
        x = randn(2)
        @inferred ef(y; σ_phys = 0.7)
        @inferred loglik(x, lik)
        @inferred loggrad(x, lik)
        @inferred loghessian(x, lik)
    end

    @testset "Routes through CompositeObservationModel still work" begin
        # Confirms aliasing on a single component is composable with composite passthrough
        m_aliased = ExponentialFamily(Normal; σ = :σ_inner)
        m_plain = ExponentialFamily(Poisson)
        composite = CompositeObservationModel((m_aliased, m_plain))

        y = CompositeObservations(([1.0, 2.0], PoissonObservations([3, 4])))
        lik = composite(y; σ_inner = 0.5)

        x = randn(2)
        ll_manual = loglik(x, m_aliased([1.0, 2.0]; σ_inner = 0.5)) +
            loglik(x, m_plain(PoissonObservations([3, 4])))
        @test loglik(x, lik) ≈ ll_manual
    end

    @testset "Show method displays aliases" begin
        ef = ExponentialFamily(Normal; σ = :σ_phys)
        s = sprint(show, ef)
        @test occursin("σ_phys", s)

        ef_plain = ExponentialFamily(Normal)
        s_plain = sprint(show, ef_plain)
        @test !occursin("alias", lowercase(s_plain))
    end
end
