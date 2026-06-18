using Test
using GaussianMarkovRandomFields
using ForwardDiff   # activates the DI ForwardDiff backend the factor assembler uses
using SparseConnectivityTracer, SparseMatrixColorings
using SparseArrays
using LinearAlgebra
using Random
using Distributions: Poisson, Normal, logpdf

@testset "StructuredObservationModel" begin

    @testset "Poisson factors, loglik/loggrad/loghessian match analytic (K=1)" begin
        Random.seed!(2)
        n = 6
        y = rand(0:4, n)
        # Family-general: the factor is just logpdf(Dist(params(vals, θ)), y_k).
        grp = ObsFactorGroup(
            [(k,) for k in 1:n], collect(1:n),
            (vals, yk, θ) -> logpdf(Poisson(exp(vals[1])), yk),
        )
        m = StructuredObservationModel(n, (grp,))
        lik = m(y)
        for _ in 1:4
            x = randn(n) .* 0.4
            @test loglik(x, lik) ≈ sum(logpdf(Poisson(exp(x[k])), y[k]) for k in 1:n) atol = 1.0e-10
            # d/dη [y·η − e^η] = y − e^η;  d²/dη² = −e^η.
            @test loggrad(x, lik) ≈ [y[k] - exp(x[k]) for k in 1:n] atol = 1.0e-8
            @test Matrix(loghessian(x, lik)) ≈ diagm([-exp(x[k]) for k in 1:n]) atol = 1.0e-8
        end
    end

    @testset "two-latent factors couple their latents (K=2)" begin
        # y[k] ~ Normal(x[k] + x[k+1], σ): each observation couples adjacent latents.
        Random.seed!(3)
        n = 5; σ = 0.7; m_obs = n - 1
        y = randn(m_obs)
        grp = ObsFactorGroup(
            [(k, k + 1) for k in 1:m_obs], collect(1:m_obs),
            (vals, yk, θ) -> logpdf(Normal(vals[1] + vals[2], σ), yk),
        )
        m = StructuredObservationModel(n, (grp,))
        lik = m(y)
        for _ in 1:4
            x = randn(n)
            r = [x[k] + x[k + 1] - y[k] for k in 1:m_obs]
            ll = sum(-0.5 * (r[k] / σ)^2 - log(σ) - 0.5 * log(2π) for k in 1:m_obs)
            @test loglik(x, lik) ≈ ll atol = 1.0e-9

            g = zeros(n)
            for k in 1:m_obs
                g[k] += -r[k] / σ^2
                g[k + 1] += -r[k] / σ^2
            end
            @test loggrad(x, lik) ≈ g atol = 1.0e-8

            H = zeros(n, n)
            for k in 1:m_obs
                H[k, k] += -1 / σ^2; H[k + 1, k + 1] += -1 / σ^2
                H[k, k + 1] += -1 / σ^2; H[k + 1, k] += -1 / σ^2
            end
            @test Matrix(loghessian(x, lik)) ≈ H atol = 1.0e-8
        end
    end

end
