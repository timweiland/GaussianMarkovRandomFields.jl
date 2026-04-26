using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: has_constraints
using Distributions
using LinearAlgebra
using SparseArrays
using Random

@testset "Workspace LatentModel Integration" begin

    @testset "GMRFWorkspace from AR1Model" begin
        n = 20
        model = AR1Model(n)
        ws = GMRFWorkspace(model; τ = 1.0, ρ = 0.5)
        @test dimension(ws) == n
    end

    @testset "GMRFWorkspace from RW1Model" begin
        n = 15
        model = RW1Model(n)
        ws = GMRFWorkspace(model; τ = 1.0)
        @test dimension(ws) == n
    end

    @testset "GMRFWorkspace from IIDModel" begin
        n = 10
        model = IIDModel(n)
        ws = GMRFWorkspace(model; τ = 2.0)
        @test dimension(ws) == n
    end

    @testset "model(ws; θ...) returns WorkspaceGMRF" begin
        n = 20
        model = AR1Model(n)
        ws = GMRFWorkspace(model; τ = 1.0, ρ = 0.5)

        prior = model(ws; τ = 2.0, ρ = 0.3)
        @test prior isa WorkspaceGMRF
        @test length(prior) == n
    end

    @testset "model(ws; θ...) matches model(; θ...)" begin
        n = 20
        model = AR1Model(n)
        ws = GMRFWorkspace(model; τ = 1.0, ρ = 0.5)

        θ = (τ = 2.0, ρ = 0.3)
        ws_gmrf = model(ws; θ...)
        ref_gmrf = model(; θ...)

        z = randn(n)
        @test logpdf(ws_gmrf, z) ≈ logpdf(ref_gmrf, z) rtol = 1.0e-8
        @test mean(ws_gmrf) ≈ mean(ref_gmrf)
        @test var(ws_gmrf) ≈ var(ref_gmrf) rtol = 1.0e-8
    end

    @testset "Repeated model(ws; θ...) calls" begin
        n = 20
        model = AR1Model(n)
        ws = GMRFWorkspace(model; τ = 1.0, ρ = 0.5)
        z = randn(n)

        for (τ, ρ) in [(1.0, 0.3), (2.0, 0.5), (0.5, 0.8)]
            ws_gmrf = model(ws; τ = τ, ρ = ρ)
            ref_gmrf = model(; τ = τ, ρ = ρ)
            @test logpdf(ws_gmrf, z) ≈ logpdf(ref_gmrf, z) rtol = 1.0e-8
        end
    end

    @testset "Constrained model(ws; θ...) returns ConstrainedWorkspaceGMRF" begin
        n = 15
        model = RW1Model(n)
        ws = GMRFWorkspace(model; τ = 1.0)

        prior = model(ws; τ = 2.0)
        @test prior isa WorkspaceGMRF
        @test has_constraints(prior)
        @test length(prior) == n
        @test sum(mean(prior)) ≈ 0.0 atol = 1.0e-8
    end

    @testset "Constrained model(ws; θ...) matches model(; θ...)" begin
        n = 15
        model = RW1Model(n)
        ws = GMRFWorkspace(model; τ = 1.0)

        ws_gmrf = model(ws; τ = 2.0)
        ref_gmrf = model(; τ = 2.0)

        @test mean(ws_gmrf) ≈ mean(ref_gmrf) rtol = 1.0e-8
        @test var(ws_gmrf) ≈ var(ref_gmrf) rtol = 1.0e-8
    end

    @testset "Full INLA-like pipeline" begin
        n = 20
        model = AR1Model(n)
        ws = GMRFWorkspace(model; τ = 1.0, ρ = 0.5)

        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations(rand(1:5, n))
        obs_lik = obs_model(y)

        # Simulate outer loop
        for (τ, ρ) in [(1.0, 0.3), (2.0, 0.5), (5.0, 0.8)]
            prior = model(ws; τ = τ, ρ = ρ)
            posterior = gaussian_approximation(prior, obs_lik)
            val = logpdf(posterior, mean(posterior))
            @test isfinite(val)

            # Verify matches reference
            ref_prior = model(; τ = τ, ρ = ρ)
            ref_posterior = gaussian_approximation(ref_prior, obs_lik)
            @test mean(posterior) ≈ mean(ref_posterior) rtol = 1.0e-6
        end
    end

    @testset "Joint pattern workspace with ObservationLikelihood" begin
        n = 20
        model = AR1Model(n)
        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations(rand(1:5, n))
        obs_lik = obs_model(y)

        # Create workspace with joint pattern (Q_prior ∪ H_obs)
        ws = GMRFWorkspace(model, obs_lik; τ = 1.0, ρ = 0.5)
        @test dimension(ws) == n

        prior = model(ws; τ = 2.0, ρ = 0.3)
        posterior = gaussian_approximation(prior, obs_lik)
        @test isfinite(logpdf(posterior, mean(posterior)))
    end

    @testset "Joint pattern workspace with non-diagonal Hessian" begin
        # Regression for the SHOULD-FIX flagged by Codex: when the obs Hessian
        # has off-diagonal nonzeros (e.g. via a design matrix), the workspace
        # pattern is strictly larger than the prior pattern. The LatentModel
        # callable must pad the prior precision into the workspace pattern,
        # otherwise update_precision!'s pattern check rejects it.
        n = 8
        m_obs = 10
        Random.seed!(123)
        A = sprand(m_obs, n, 0.6)  # non-square design matrix
        model = AR1Model(n)
        base = ExponentialFamily(Distributions.Normal)
        ltom = LinearlyTransformedObservationModel(base, A)
        y_obs = randn(m_obs)
        obs_lik = ltom(y_obs; σ = 0.5)

        ws = GMRFWorkspace(model, obs_lik; τ = 1.0, ρ = 0.5)

        # Workspace pattern is strictly a superset of the prior pattern.
        Q_prior_only = sparse(precision_matrix(model; τ = 1.0, ρ = 0.5))
        @test nnz(ws.Q) > nnz(Q_prior_only)

        # The LatentModel callable must accept this configuration without
        # tripping the pattern-equality check inside update_precision!.
        prior = model(ws; τ = 2.0, ρ = 0.3)
        @test prior isa WorkspaceGMRF

        # Numerical correctness against a fresh-construction baseline.
        ref_prior = model(; τ = 2.0, ρ = 0.3)
        z = randn(n)
        @test logpdf(prior, z) ≈ logpdf(ref_prior, z) rtol = 1.0e-8

        posterior = gaussian_approximation(prior, obs_lik)
        @test isfinite(logpdf(posterior, mean(posterior)))
    end
end
