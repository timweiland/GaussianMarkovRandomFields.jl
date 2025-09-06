using Test
using GaussianMarkovRandomFields
using Distributions

@testset "NonlinearLeastSquares" begin
    # f: R^2 -> R^3
    f = x -> begin
        @assert length(x) == 2
        return [x[1] + 2x[2], sin(x[1]), x[2]^2]
    end

    # Analytical Jacobian
    function J_analytic(x)
        @assert length(x) == 2
        return [
            1.0 2.0;
            cos(x[1]) 0.0;
            0.0 2x[2]
        ]
    end

    @testset "Basic (scalar sigma)" begin
        n = 2
        y = [1.0, 0.5, 2.0]
        σ = 0.3
        x = [0.1, -0.2]

        model = NonlinearLeastSquaresModel(f, n)
        lik = model(y; σ = σ)

        # Manual computations
        yhat = f(x)
        r = y .- yhat
        J = J_analytic(x)
        w = 1 / σ^2
        log_const = -0.5 * length(y) * log(2π) - length(y) * log(σ)

        ll_expected = log_const - 0.5 * w * sum(r .^ 2)
        g_expected = J' * (w .* r)
        H_expected = -(J' * (Diagonal(fill(w, length(y))) * J))

        @test loglik(x, lik) ≈ ll_expected atol = 1.0e-10
        @test loggrad(x, lik) ≈ g_expected atol = 1.0e-8
        @test loghessian(x, lik) ≈ Symmetric(H_expected) atol = 1.0e-8
    end

    @testset "Heteroskedastic (vector sigma)" begin
        n = 2
        y = [1.0, 0.5, 2.0]
        σ = [0.4, 0.5, 0.6]
        x = [0.3, 0.1]

        model = NonlinearLeastSquaresModel(f, n)
        lik = model(y; σ = σ)

        yhat = f(x)
        r = y .- yhat
        J = J_analytic(x)
        w = 1.0 ./ (σ .^ 2)
        W = Diagonal(w)
        log_const = -0.5 * length(y) * log(2π) - sum(log, σ)

        ll_expected = log_const - 0.5 * sum(w .* (r .^ 2))
        g_expected = J' * (w .* r)
        H_expected = -(J' * W * J)

        @test loglik(x, lik) ≈ ll_expected atol = 1.0e-10
        @test loggrad(x, lik) ≈ g_expected atol = 1.0e-8
        @test loghessian(x, lik) ≈ Symmetric(H_expected) atol = 1.0e-8
    end

    @testset "Dimension validation" begin
        # y length inconsistent with f(x) dimension
        model = NonlinearLeastSquaresModel(f, 2)
        y_bad = [1.0, 0.5]  # length 2, but f returns length 3
        lik_bad = model(y_bad; σ = 0.2)
        @test_throws DimensionMismatch loglik(zeros(2), lik_bad)
    end

    @testset "Interface hooks" begin
        n = 2
        model = NonlinearLeastSquaresModel(f, n)
        y = [1.0, 0.5, 2.0]
        @test hyperparameters(model) == (:σ,)
        @test latent_dimension(model, y) == n
    end

    @testset "Conditional distribution" begin
        model = NonlinearLeastSquaresModel(f, 2)
        x = [0.2, -0.1]
        σ = 0.7
        d = conditional_distribution(model, x; σ = σ)
        yhat = f(x)
        y_sample = yhat .+ σ .* randn(length(yhat))
        # Compare logpdfs against manual product of Normals
        manual = sum(logpdf.(Normal.(yhat, σ), y_sample))
        @test logpdf(d, y_sample) ≈ manual atol = 1.0e-10
    end
end
