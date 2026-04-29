using Distributions
using ForwardDiff
using LinearAlgebra
using StatsFuns
using Random
using StatsModels
using SparseArrays

@testset "Negative Binomial Observation Model" begin
    # ============================================================
    # NegativeBinomialObservations
    # ============================================================
    @testset "NegativeBinomialObservations" begin
        @testset "Construction with exposure" begin
            y = NegativeBinomialObservations([3, 1, 8], [1.0, 2.5, 0.75])
            @test length(y) == 3
            @test counts(y) == [3, 1, 8]
            @test exposure(y) ≈ [1.0, 2.5, 0.75]
            @test GaussianMarkovRandomFields.logexposure(y) ≈ log.([1.0, 2.5, 0.75])
        end

        @testset "Construction without exposure" begin
            y = NegativeBinomialObservations([5, 0, 2])
            @test length(y) == 3
            @test counts(y) == [5, 0, 2]
            @test exposure(y) ≈ [1.0, 1.0, 1.0]
            @test GaussianMarkovRandomFields.logexposure(y) ≈ [0.0, 0.0, 0.0]
        end

        @testset "Tuple indexing" begin
            y = NegativeBinomialObservations([3, 1], [2.0, 0.5])
            @test y[1] == (3, 2.0)
            @test y[2] == (1, 0.5)
        end

        @testset "Range indexing and iteration" begin
            y = NegativeBinomialObservations([3, 1, 8], [2.0, 0.5, 1.0])
            @test y[1:2] == [(3, 2.0), (1, 0.5)]
            collected = collect(y)
            @test collected == [(3, 2.0), (1, 0.5), (8, 1.0)]
        end

        @testset "Property access" begin
            y = NegativeBinomialObservations([3, 1], [2.0, 0.5])
            @test y.exposure ≈ [2.0, 0.5]
            @test y.counts == [3, 1]
        end

        @testset "Validation" begin
            @test_throws ErrorException NegativeBinomialObservations([-1, 2])
            @test_throws ErrorException NegativeBinomialObservations([1, 2], [0.0, 1.0])
            @test_throws ErrorException NegativeBinomialObservations([1, 2], [-1.0, 1.0])
            @test_throws ErrorException NegativeBinomialObservations([1], [1.0, 2.0])
        end
    end

    # ============================================================
    # ExponentialFamily construction
    # ============================================================
    @testset "ExponentialFamily construction" begin
        ef = ExponentialFamily(NegativeBinomial)
        @test ef.link isa LogLink
        @test hyperparameters(ef) == (:r,)
    end

    # ============================================================
    # Likelihood materialization
    # ============================================================
    @testset "Likelihood materialization" begin
        ef = ExponentialFamily(NegativeBinomial)
        y = NegativeBinomialObservations([3, 1, 8])
        lik = ef(y; r = 5.0)
        @test lik isa NegBinLikelihood
        @test lik.r == 5.0
        @test lik.y == [3, 1, 8]
    end

    # ============================================================
    # LogLink (canonical-like): loglik, gradient, Hessian
    # ============================================================
    @testset "LogLink correctness" begin
        ef = ExponentialFamily(NegativeBinomial)
        count_data = [3, 1, 8, 0, 5]
        y = NegativeBinomialObservations(count_data)
        r = 5.0
        lik = ef(y; r = r)
        η = randn(5)

        # loglik matches Distributions.jl reference
        μ = exp.(η)
        p = r ./ (r .+ μ)
        ref_ll = sum(logpdf.(NegativeBinomial.(r, p), count_data))
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
    # LogLink with exposure/offset
    # ============================================================
    @testset "LogLink with exposure" begin
        ef = ExponentialFamily(NegativeBinomial)
        count_data = [3, 1, 8]
        exp_data = [1.0, 2.5, 0.75]
        y = NegativeBinomialObservations(count_data, exp_data)
        r = 3.0
        lik = ef(y; r = r)
        η = randn(3)

        # loglik with exposure: μ = exp(η + log(exposure))
        μ = exp.(η) .* exp_data
        p = r ./ (r .+ μ)
        ref_ll = sum(logpdf.(NegativeBinomial.(r, p), count_data))
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
        ef = ExponentialFamily(NegativeBinomial, IdentityLink())
        count_data = [3, 1, 8]
        y = NegativeBinomialObservations(count_data)
        r = 5.0
        lik = ef(y; r = r)
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
        ef = ExponentialFamily(NegativeBinomial; indices = 2:4)
        count_data = [3, 1, 8]
        y = NegativeBinomialObservations(count_data)
        r = 5.0
        lik = ef(y; r = r)
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
        ef = ExponentialFamily(NegativeBinomial)
        count_data = [3, 1, 8, 0, 5]
        y = NegativeBinomialObservations(count_data)
        r = 5.0
        lik = ef(y; r = r)
        η = randn(5)

        pw = pointwise_loglik(η, lik)
        @test length(pw) == 5

        # Each element matches scalar logpdf
        μ = exp.(η)
        p = r ./ (r .+ μ)
        for i in 1:5
            @test pw[i] ≈ logpdf(NegativeBinomial(r, p[i]), count_data[i]) atol = 1.0e-10
        end

        # Sum matches total loglik
        @test sum(pw) ≈ loglik(η, lik) atol = 1.0e-10

        # In-place version
        result = zeros(5)
        pointwise_loglik!(result, η, lik)
        @test result ≈ pw atol = 1.0e-10
    end

    # ============================================================
    # Poisson limit: as r → ∞, NB → Poisson
    # ============================================================
    @testset "Poisson limit (r → ∞)" begin
        count_data = [3, 1, 8, 0, 5]
        Random.seed!(42)
        η = randn(5)
        r_large = 1.0e8

        # NB likelihood
        ef_nb = ExponentialFamily(NegativeBinomial)
        y_nb = NegativeBinomialObservations(count_data)
        lik_nb = ef_nb(y_nb; r = r_large)

        # Poisson likelihood
        ef_pois = ExponentialFamily(Poisson)
        y_pois = PoissonObservations(count_data)
        lik_pois = ef_pois(y_pois)

        # Gradient should converge
        grad_nb = loggrad(η, lik_nb)
        grad_pois = loggrad(η, lik_pois)
        @test grad_nb ≈ grad_pois rtol = 1.0e-6

        # Hessian should converge
        hess_nb = loghessian(η, lik_nb)
        hess_pois = loghessian(η, lik_pois)
        @test Matrix(hess_nb) ≈ Matrix(hess_pois) rtol = 1.0e-6
    end

    # ============================================================
    # conditional_distribution (sampling)
    # ============================================================
    @testset "conditional_distribution" begin
        ef = ExponentialFamily(NegativeBinomial)
        η = [1.0, 0.5, -0.5]
        dist = conditional_distribution(ef, η; r = 5.0)
        samples = rand(dist)
        @test length(samples) == 3
        @test all(s -> s >= 0, samples)

        # With offset (log-exposure)
        offset = log.([2.0, 3.0, 1.5])
        dist_off = conditional_distribution(ef, η; r = 5.0, offset = offset)
        samples_off = rand(dist_off)
        @test length(samples_off) == 3
        @test all(s -> s >= 0, samples_off)

        # Offset with non-LogLink should error
        ef_id = ExponentialFamily(NegativeBinomial, IdentityLink())
        @test_throws ArgumentError conditional_distribution(ef_id, abs.(η) .+ 1.0; r = 5.0, offset = [0.1, 0.2, 0.3])
    end

    # ============================================================
    # Formula interface
    # ============================================================
    @testset "Formula interface" begin
        n = 10
        data = (
            y = rand(0:10, n),
            group = repeat(1:2, inner = div(n, 2)),
        )

        iid = IID()
        comp = build_formula_components(@formula(y ~ 0 + iid(group)), data; family = NegativeBinomial)
        @test comp.y isa NegativeBinomialObservations
        @test counts(comp.y) == data.y

        # With exposure
        data_exp = (
            y = rand(0:10, n),
            group = repeat(1:2, inner = div(n, 2)),
            pop = rand(100:1000, n),
        )
        comp_exp = build_formula_components(
            @formula(y ~ 0 + iid(group)), data_exp;
            family = NegativeBinomial, exposure = :pop,
        )
        @test comp_exp.y isa NegativeBinomialObservations
        @test exposure(comp_exp.y) ≈ Float64.(data_exp.pop)
    end
end
