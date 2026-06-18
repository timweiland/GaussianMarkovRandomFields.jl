using ReTest: @testset, @test
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

    @testset "pointwise_loglik, accessors, and multiple factor groups" begin
        # Two groups with different factor forms (Poisson on x[1:3], Normal on x[4:5] with a σ
        # hyperparameter), exercising the heterogeneous-group recursion, the accessors, and the
        # per-factor pointwise log-likelihood.
        Random.seed!(5)
        n = 5
        y = [1.0, 0.0, 2.0, 0.3, -0.5]   # y[1:3]: Poisson counts; y[4:5]: Normal observations
        g_pois = ObsFactorGroup(
            [(k,) for k in 1:3], [1, 2, 3],
            (vals, yk, θ) -> logpdf(Poisson(exp(vals[1])), yk),
        )
        g_norm = ObsFactorGroup(
            [(4,), (5,)], [4, 5],
            (vals, yk, θ) -> logpdf(Normal(vals[1], θ.σ), yk),
        )
        m = StructuredObservationModel(n, (g_pois, g_norm); hyperparams = (:σ,))

        @test latent_dimension(m, y) == n
        @test hyperparameters(m) == (:σ,)

        lik = m(y; σ = 0.7)
        x = randn(n) .* 0.3
        pw = pointwise_loglik(x, lik)
        @test length(pw) == 5            # 3 Poisson + 2 Normal factors
        @test sum(pw) ≈ loglik(x, lik) atol = 1.0e-10
        # The per-factor values match direct evaluation of each factor.
        @test pw[1] ≈ logpdf(Poisson(exp(x[1])), y[1]) atol = 1.0e-10
        @test pw[4] ≈ logpdf(Normal(x[4], 0.7), y[4]) atol = 1.0e-10
    end

end
