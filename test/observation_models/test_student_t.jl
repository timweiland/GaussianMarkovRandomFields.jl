using Distributions
using ForwardDiff
using LinearAlgebra
using StatsFuns
using Random
using StatsModels
using SparseArrays

@testset "Student-t Observation Model" begin
    # ============================================================
    # ExponentialFamily construction
    # ============================================================
    @testset "ExponentialFamily construction" begin
        ef = ExponentialFamily(TDist)
        @test ef.link isa IdentityLink
        @test hyperparameters(ef) == (:σ, :ν)
    end

    # ============================================================
    # Likelihood materialization
    # ============================================================
    @testset "Likelihood materialization" begin
        ef = ExponentialFamily(TDist)
        y = [1.5, -0.3, 4.2]
        lik = ef(y; σ = 1.0, ν = 4.0)
        @test lik isa StudentTLikelihood
        @test lik.σ == 1.0
        @test lik.ν == 4.0
        @test lik.y == [1.5, -0.3, 4.2]
    end

    @testset "Validation" begin
        ef = ExponentialFamily(TDist)
        @test_throws ErrorException ef([1.0, 2.0]; σ = 0.0, ν = 4.0)
        @test_throws ErrorException ef([1.0, 2.0]; σ = -1.0, ν = 4.0)
        @test_throws ErrorException ef([1.0, 2.0]; σ = 1.0, ν = 2.0)
        @test_throws ErrorException ef([1.0, 2.0]; σ = 1.0, ν = 1.5)
    end

    # ============================================================
    # IdentityLink (canonical): loglik, gradient, Hessian
    # ============================================================
    @testset "IdentityLink correctness" begin
        ef = ExponentialFamily(TDist)
        y_data = [1.5, -0.3, 4.2, 0.8, -2.1]
        σ = 2.0
        ν = 5.0
        lik = ef(y_data; σ = σ, ν = ν)
        η = randn(5)

        # loglik matches Distributions.jl reference
        σ_eff = σ * sqrt((ν - 2) / ν)
        ref_ll = sum(logpdf.(η .+ σ_eff .* TDist(ν), y_data))
        @test loglik(η, lik) ≈ ref_ll atol = 1.0e-10

        # Gradient matches ForwardDiff
        grad = loggrad(η, lik)
        grad_fd = ForwardDiff.gradient(x -> loglik(x, lik), η)
        @test grad ≈ grad_fd atol = 1.0e-8

        # Hessian matches ForwardDiff
        hess = loghessian(η, lik)
        hess_fd = ForwardDiff.hessian(x -> loglik(x, lik), η)
        @test Matrix(hess) ≈ hess_fd atol = 1.0e-6
    end

    # ============================================================
    # Non-canonical link (LogLink via chain rule fallback)
    # ============================================================
    @testset "LogLink (non-canonical)" begin
        ef = ExponentialFamily(TDist, LogLink())
        y_data = [1.5, 0.3, 4.2]
        σ = 2.0
        ν = 5.0
        lik = ef(y_data; σ = σ, ν = ν)
        η = randn(3)

        # Gradient matches ForwardDiff
        grad = loggrad(η, lik)
        grad_fd = ForwardDiff.gradient(x -> loglik(x, lik), η)
        @test grad ≈ grad_fd atol = 1.0e-6

        # Hessian matches ForwardDiff
        hess = loghessian(η, lik)
        hess_fd = ForwardDiff.hessian(x -> loglik(x, lik), η)
        @test Matrix(hess) ≈ hess_fd atol = 1.0e-5
    end

    # ============================================================
    # Indexed observations
    # ============================================================
    @testset "Indexed observations" begin
        ef = ExponentialFamily(TDist; indices = 2:4)
        y_data = [1.5, -0.3, 4.2]
        σ = 2.0
        ν = 5.0
        lik = ef(y_data; σ = σ, ν = ν)
        η_full = randn(6)

        # Gradient should be zero outside indices
        grad = loggrad(η_full, lik)
        @test grad[1] == 0.0
        @test grad[5] == 0.0
        @test grad[6] == 0.0
        @test any(grad[2:4] .!= 0.0)

        # Matches ForwardDiff
        grad_fd = ForwardDiff.gradient(x -> loglik(x, lik), η_full)
        @test grad ≈ grad_fd atol = 1.0e-8
    end

    # ============================================================
    # Pointwise loglik
    # ============================================================
    @testset "Pointwise loglik" begin
        ef = ExponentialFamily(TDist)
        y_data = [1.5, -0.3, 4.2, 0.8, -2.1]
        σ = 2.0
        ν = 5.0
        lik = ef(y_data; σ = σ, ν = ν)
        η = randn(5)

        pw = pointwise_loglik(η, lik)
        @test length(pw) == 5

        # Each element matches scalar logpdf
        σ_eff = σ * sqrt((ν - 2) / ν)
        for i in 1:5
            @test pw[i] ≈ logpdf(η[i] + σ_eff * TDist(ν), y_data[i]) atol = 1.0e-10
        end

        # Sum matches total loglik
        @test sum(pw) ≈ loglik(η, lik) atol = 1.0e-10

        # In-place version
        result = zeros(5)
        pointwise_loglik!(result, η, lik)
        @test result ≈ pw atol = 1.0e-10
    end

    # ============================================================
    # Normal limit: as ν → ∞, converge to Normal
    # ============================================================
    @testset "Normal limit" begin
        y_data = [1.5, -0.3, 4.2, 0.8, -2.1]
        σ = 2.0
        η = randn(5)

        # Student-t with large ν
        ef_t = ExponentialFamily(TDist)
        lik_t = ef_t(y_data; σ = σ, ν = 1.0e6)

        # Normal reference
        ef_n = ExponentialFamily(Normal)
        lik_n = ef_n(y_data; σ = σ)

        # Gradient should converge
        grad_t = loggrad(η, lik_t)
        grad_n = loggrad(η, lik_n)
        @test grad_t ≈ grad_n atol = 1.0e-4

        # Hessian should converge
        hess_t = loghessian(η, lik_t)
        hess_n = loghessian(η, lik_n)
        @test Matrix(hess_t) ≈ Matrix(hess_n) atol = 1.0e-4
    end

    # ============================================================
    # conditional_distribution (sampling)
    # ============================================================
    @testset "conditional_distribution" begin
        ef = ExponentialFamily(TDist)
        η = [1.0, 0.5, -0.5]
        dist = conditional_distribution(ef, η; σ = 2.0, ν = 4.0)
        samples = rand(dist)
        @test length(samples) == 3
    end

    # ============================================================
    # Formula interface
    # ============================================================
    @testset "Formula interface" begin
        n = 10
        Random.seed!(42)
        data = (
            y = randn(n),
            group = repeat(1:2, inner = div(n, 2)),
        )

        iid = IID()
        comp = build_formula_components(@formula(y ~ 0 + iid(group)), data; family = TDist)
        @test comp.y isa Vector{Float64}
        @test comp.y == data.y
    end
end
