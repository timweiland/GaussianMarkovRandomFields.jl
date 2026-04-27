using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays

@testset "Workspace Gaussian Approximation" begin

    @testset "Poisson likelihood - matches GMRF GA" begin
        n = 10
        # AR1-like precision
        dv = 2.0 * ones(n)
        ev = fill(-0.8, n - 1)
        Q_prior = spdiagm(0 => dv, 1 => ev, -1 => ev)
        μ_prior = zeros(n)

        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations([2, 1, 3, 0, 4, 1, 2, 3, 1, 0])
        obs_lik = obs_model(y)

        # Reference: regular GMRF GA
        ref_gmrf = GMRF(μ_prior, Q_prior)
        ref_result = gaussian_approximation(ref_gmrf, obs_lik)

        # Workspace GA
        ws_gmrf = WorkspaceGMRF(μ_prior, Q_prior)
        ws_result = gaussian_approximation(ws_gmrf, obs_lik)

        @test ws_result isa WorkspaceGMRF
        @test mean(ws_result) ≈ mean(ref_result) rtol = 1.0e-8
        @test precision_matrix(ws_result) ≈ precision_matrix(ref_result) rtol = 1.0e-8
        @test logpdf(ws_result, mean(ws_result)) ≈ logpdf(ref_result, mean(ref_result)) rtol = 1.0e-6
    end

    @testset "Bernoulli likelihood - matches GMRF GA" begin
        n = 8
        Q_prior = spdiagm(0 => 2.0 * ones(n), 1 => fill(-0.5, n - 1), -1 => fill(-0.5, n - 1))
        μ_prior = zeros(n)

        obs_model = ExponentialFamily(Distributions.Bernoulli)
        y = [1, 1, 0, 1, 0, 0, 1, 0]
        obs_lik = obs_model(y)

        ref_gmrf = GMRF(μ_prior, Q_prior)
        ref_result = gaussian_approximation(ref_gmrf, obs_lik)

        ws_gmrf = WorkspaceGMRF(μ_prior, Q_prior)
        ws_result = gaussian_approximation(ws_gmrf, obs_lik)

        @test ws_result isa WorkspaceGMRF
        @test mean(ws_result) ≈ mean(ref_result) rtol = 1.0e-6
        @test precision_matrix(ws_result) ≈ precision_matrix(ref_result) rtol = 1.0e-6
    end

    @testset "Gaussian likelihood - conjugate dispatch still works" begin
        n = 5
        Q_prior = spdiagm(0 => 2.0 * ones(n))
        μ_prior = zeros(n)

        obs_model = ExponentialFamily(Distributions.Normal)
        y = [0.1, 0.2, -0.1, 0.3, -0.2]
        obs_lik = obs_model(y; σ = 0.5)

        ref_gmrf = GMRF(μ_prior, Q_prior)
        ref_result = gaussian_approximation(ref_gmrf, obs_lik)

        # For Normal likelihood the existing code dispatches to linear_condition.
        # We don't need a special workspace path for conjugate cases,
        # but it shouldn't error. A WorkspaceGMRF can fall back to the
        # non-workspace GA path or we provide a specific method.
        ws_gmrf = WorkspaceGMRF(μ_prior, Q_prior)
        ws_result = gaussian_approximation(ws_gmrf, obs_lik)

        @test mean(ws_result) ≈ mean(ref_result) rtol = 1.0e-8
    end

    @testset "Posterior workspace holds Q_post factorization" begin
        n = 10
        Q_prior = spdiagm(0 => 2.0 * ones(n), 1 => fill(-0.8, n - 1), -1 => fill(-0.8, n - 1))
        μ_prior = zeros(n)

        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations([2, 1, 3, 0, 4, 1, 2, 3, 1, 0])
        obs_lik = obs_model(y)

        ws_gmrf = WorkspaceGMRF(μ_prior, Q_prior)
        ws_result = gaussian_approximation(ws_gmrf, obs_lik)

        # The workspace should hold Q_post's factorization, so var/logdet should work
        v = var(ws_result)
        @test all(v .> 0)
        @test length(v) == n

        ld = logdetcov(ws_result)
        @test isfinite(ld)
    end

    @testset "Prior precision is preserved after GA" begin
        n = 10
        Q_prior = spdiagm(0 => 2.0 * ones(n), 1 => fill(-0.8, n - 1), -1 => fill(-0.8, n - 1))
        μ_prior = zeros(n)

        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations([2, 1, 3, 0, 4, 1, 2, 3, 1, 0])
        obs_lik = obs_model(y)

        ws_gmrf = WorkspaceGMRF(μ_prior, Q_prior)
        Q_prior_copy = copy(Q_prior)

        ws_result = gaussian_approximation(ws_gmrf, obs_lik)

        # Prior's owned Q should be untouched
        @test precision_matrix(ws_gmrf) == Q_prior_copy
        # Posterior's Q should be different (Q_prior + observation info)
        @test precision_matrix(ws_result) != Q_prior_copy
    end

    @testset "Repeated GA calls with workspace reuse" begin
        n = 10
        Q_prior = spdiagm(0 => 2.0 * ones(n), 1 => fill(-0.8, n - 1), -1 => fill(-0.8, n - 1))
        μ_prior = zeros(n)
        ws = GMRFWorkspace(Q_prior)

        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations([2, 1, 3, 0, 4, 1, 2, 3, 1, 0])
        obs_lik = obs_model(y)

        # Simulate outer loop: different "hyperparameters" (scale Q)
        results = []
        for scale in [1.0, 2.0, 0.5]
            Q_scaled = Q_prior * scale
            update_precision!(ws, Q_scaled)
            prior = WorkspaceGMRF(μ_prior, Q_scaled, ws)
            posterior = gaussian_approximation(prior, obs_lik)
            push!(results, (mode = copy(mean(posterior)), Q = copy(precision_matrix(posterior))))
        end

        # Verify each matches the reference
        for (i, scale) in enumerate([1.0, 2.0, 0.5])
            Q_scaled = Q_prior * scale
            ref = gaussian_approximation(GMRF(μ_prior, Q_scaled), obs_lik)
            @test results[i].mode ≈ mean(ref) rtol = 1.0e-6
            @test results[i].Q ≈ precision_matrix(ref) rtol = 1.0e-6
        end
    end

    @testset "Non-diagonal Hessian (sparse)" begin
        # Test with a sparse observation Hessian by using indexed observations
        # where only a subset of the latent field is observed
        n = 10
        Q_prior = spdiagm(0 => 2.0 * ones(n), 1 => fill(-0.8, n - 1), -1 => fill(-0.8, n - 1))
        μ_prior = zeros(n)

        # Observe only indices 1, 3, 5, 7, 9
        obs_indices = [1, 3, 5, 7, 9]
        obs_model = ExponentialFamily(Distributions.Poisson; indices = obs_indices)
        y = PoissonObservations([2, 3, 4, 1, 2])
        obs_lik = obs_model(y)

        ref_gmrf = GMRF(μ_prior, Q_prior)
        ref_result = gaussian_approximation(ref_gmrf, obs_lik)

        ws_gmrf = WorkspaceGMRF(μ_prior, Q_prior)
        ws_result = gaussian_approximation(ws_gmrf, obs_lik)

        @test mean(ws_result) ≈ mean(ref_result) rtol = 1.0e-6
        @test precision_matrix(ws_result) ≈ precision_matrix(ref_result) rtol = 1.0e-6
    end
end
