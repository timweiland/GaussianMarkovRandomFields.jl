using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: has_constraints
using Distributions
using LinearAlgebra
using SparseArrays

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
end
