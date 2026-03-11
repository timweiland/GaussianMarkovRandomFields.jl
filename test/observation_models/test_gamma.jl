using Distributions
using ForwardDiff
using LinearAlgebra
using StatsFuns
using Random
using StatsModels
using SparseArrays

@testset "Gamma Observation Model" begin
    # ============================================================
    # ExponentialFamily construction
    # ============================================================
    @testset "ExponentialFamily construction" begin
        ef = ExponentialFamily(Gamma)
        @test ef.link isa LogLink
        @test hyperparameters(ef) == (:phi,)
    end

    # ============================================================
    # Likelihood materialization
    # ============================================================
    @testset "Likelihood materialization" begin
        ef = ExponentialFamily(Gamma)
        y = [1.5, 0.3, 4.2]
        lik = ef(y; phi = 2.0)
        @test lik isa GammaLikelihood
        @test lik.phi == 2.0
        @test lik.y == [1.5, 0.3, 4.2]
    end

    @testset "Validation" begin
        ef = ExponentialFamily(Gamma)
        @test_throws ErrorException ef([1.0, -0.5, 2.0]; phi = 1.0)
        @test_throws ErrorException ef([1.0, 0.0, 2.0]; phi = 1.0)
        @test_throws ErrorException ef([1.0, 2.0]; phi = 0.0)
        @test_throws ErrorException ef([1.0, 2.0]; phi = -1.0)
    end

    # ============================================================
    # LogLink (canonical-like): loglik, gradient, Hessian
    # ============================================================
    @testset "LogLink correctness" begin
        ef = ExponentialFamily(Gamma)
        y_data = [1.5, 0.3, 4.2, 0.8, 2.1]
        phi = 3.0
        lik = ef(y_data; phi = phi)
        η = randn(5)

        # loglik matches Distributions.jl reference
        μ = exp.(η)
        # Distributions.Gamma(α, θ) uses shape α and scale θ
        # Our parameterization: shape = φ, scale = μ/φ
        ref_ll = sum(logpdf.(Gamma.(phi, μ ./ phi), y_data))
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
    # Non-canonical link (IdentityLink via chain rule fallback)
    # ============================================================
    @testset "IdentityLink (non-canonical)" begin
        ef = ExponentialFamily(Gamma, IdentityLink())
        y_data = [1.5, 0.3, 4.2]
        phi = 3.0
        lik = ef(y_data; phi = phi)
        # η must be positive for IdentityLink (μ = η)
        η = abs.(randn(3)) .+ 1.0

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
        ef = ExponentialFamily(Gamma; indices = 2:4)
        y_data = [1.5, 0.3, 4.2]
        phi = 3.0
        lik = ef(y_data; phi = phi)
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
        ef = ExponentialFamily(Gamma)
        y_data = [1.5, 0.3, 4.2, 0.8, 2.1]
        phi = 3.0
        lik = ef(y_data; phi = phi)
        η = randn(5)

        pw = pointwise_loglik(η, lik)
        @test length(pw) == 5

        # Each element matches scalar logpdf
        μ = exp.(η)
        for i in 1:5
            @test pw[i] ≈ logpdf(Gamma(phi, μ[i] / phi), y_data[i]) atol = 1.0e-10
        end

        # Sum matches total loglik
        @test sum(pw) ≈ loglik(η, lik) atol = 1.0e-10

        # In-place version
        result = zeros(5)
        pointwise_loglik!(result, η, lik)
        @test result ≈ pw atol = 1.0e-10
    end

    # ============================================================
    # conditional_distribution (sampling)
    # ============================================================
    @testset "conditional_distribution" begin
        ef = ExponentialFamily(Gamma)
        η = [1.0, 0.5, -0.5]
        dist = conditional_distribution(ef, η; phi = 3.0)
        samples = rand(dist)
        @test length(samples) == 3
        @test all(s -> s > 0, samples)
    end

    # ============================================================
    # Formula interface
    # ============================================================
    @testset "Formula interface" begin
        n = 10
        Random.seed!(42)
        data = (
            y = rand(Gamma(2.0, 1.0), n),
            group = repeat(1:2, inner = div(n, 2)),
        )

        iid = IID()
        comp = build_formula_components(@formula(y ~ 0 + iid(group)), data; family = Gamma)
        @test comp.y isa Vector{Float64}
        @test comp.y == data.y
    end
end
