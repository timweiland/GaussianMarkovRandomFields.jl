using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: workspace_solve, backward_solve, selinv, selinv_diag
using Distributions: logpdf
using LinearAlgebra
using SparseArrays
using Random

@testset "WorkspaceGMRF" begin
    n = 20
    Q = _make_test_precision(n)
    Q_dense = Matrix(Q)
    Q_inv = inv(Q_dense)
    μ = randn(n)

    # Reference GMRF for comparison
    ref_gmrf = GMRF(μ, Q)

    @testset "Construction from Q" begin
        wg = WorkspaceGMRF(μ, Q)
        @test length(wg) == n
        @test mean(wg) == μ
        @test precision_matrix(wg) == Q
        @test precision_map(wg) == Q
    end

    @testset "Construction from existing workspace" begin
        ws = GMRFWorkspace(Q)
        wg = WorkspaceGMRF(μ, Q, ws)
        @test length(wg) == n
        @test mean(wg) == μ
        @test precision_matrix(wg) == Q  # snapshot (copy)
        @test wg.workspace === ws  # same workspace
    end

    @testset "logpdf matches GMRF" begin
        wg = WorkspaceGMRF(μ, Q)
        z = randn(n)
        @test logpdf(wg, z) ≈ logpdf(ref_gmrf, z) rtol = 1.0e-10
    end

    @testset "logpdf at mean" begin
        wg = WorkspaceGMRF(μ, Q)
        # At the mean, quadratic form is zero, so logpdf = 0.5 * logdet(Q) - 0.5 * n * log(2π)
        expected = 0.5 * logdet(Q_dense) - 0.5 * n * log(2π)
        @test logpdf(wg, μ) ≈ expected rtol = 1.0e-10
    end

    @testset "logdetcov matches GMRF" begin
        wg = WorkspaceGMRF(μ, Q)
        @test logdetcov(wg) ≈ logdetcov(ref_gmrf) rtol = 1.0e-10
    end

    @testset "var matches GMRF" begin
        wg = WorkspaceGMRF(μ, Q)
        @test var(wg) ≈ var(ref_gmrf) rtol = 1.0e-8
    end

    @testset "std matches GMRF" begin
        wg = WorkspaceGMRF(μ, Q)
        @test std(wg) ≈ std(ref_gmrf) rtol = 1.0e-8
    end

    @testset "Sampling" begin
        wg = WorkspaceGMRF(μ, Q)
        rng = Random.MersenneTwister(42)
        n_samples = 50000
        samples = rand(rng, wg, n_samples)

        empirical_mean = mean(samples, dims = 2)[:, 1]
        @test empirical_mean ≈ μ rtol = 0.15

        empirical_var = var(samples, dims = 2)[:, 1]
        @test empirical_var ≈ diag(Q_inv) rtol = 0.15
    end

    @testset "Squared Mahalanobis distance" begin
        wg = WorkspaceGMRF(μ, Q)
        x = randn(n)
        @test sqmahal(wg, x) ≈ sqmahal(ref_gmrf, x) rtol = 1.0e-10
        @test sqmahal(wg, μ) ≈ 0.0 atol = 1.0e-12
    end

    @testset "gradlogpdf" begin
        wg = WorkspaceGMRF(μ, Q)
        @test gradlogpdf(wg, μ) ≈ zeros(n) atol = 1.0e-10
        x = randn(n)
        @test gradlogpdf(wg, x) ≈ gradlogpdf(ref_gmrf, x) rtol = 1.0e-10
    end

    @testset "Workspace reuse across updates" begin
        ws = GMRFWorkspace(Q)

        # First WorkspaceGMRF
        wg1 = WorkspaceGMRF(μ, Q, ws)
        z = randn(n)
        val1 = logpdf(wg1, z)

        # Update workspace with scaled Q
        Q2 = copy(Q)
        Q2.nzval .*= 2.0
        update_precision!(ws, Q2)

        # Second WorkspaceGMRF with different mean and Q
        μ2 = randn(n)
        wg2 = WorkspaceGMRF(μ2, Q2, ws)

        # wg2 should use the updated workspace
        ref2 = GMRF(μ2, Q2)
        @test logpdf(wg2, z) ≈ logpdf(ref2, z) rtol = 1.0e-10
        @test var(wg2) ≈ var(ref2) rtol = 1.0e-8

        # wg1's precision is preserved (owns its own Q)
        @test precision_matrix(wg1) == Q
        @test precision_matrix(wg1) != Q2
    end

    @testset "Version coherence across shared workspace" begin
        ws = GMRFWorkspace(Q)

        # Two WorkspaceGMRFs with different Q, sharing a workspace
        Q2 = copy(Q)
        Q2.nzval .*= 2.0
        μ2 = randn(n)

        wg1 = WorkspaceGMRF(μ, Q, ws)
        wg2 = WorkspaceGMRF(μ2, Q2, ws)

        z = randn(n)
        ref1 = GMRF(μ, Q)
        ref2 = GMRF(μ2, Q2)

        # Using wg2 last, workspace holds Q2's factorization
        @test logpdf(wg2, z) ≈ logpdf(ref2, z) rtol = 1.0e-10
        @test var(wg2) ≈ var(ref2) rtol = 1.0e-8

        # Now use wg1 — ensure_loaded! should reload Q1 and refactorize
        @test logpdf(wg1, z) ≈ logpdf(ref1, z) rtol = 1.0e-10
        @test var(wg1) ≈ var(ref1) rtol = 1.0e-8

        # Ping-pong back to wg2
        @test logpdf(wg2, z) ≈ logpdf(ref2, z) rtol = 1.0e-10

        # And back to wg1
        @test logpdf(wg1, z) ≈ logpdf(ref1, z) rtol = 1.0e-10
    end

    @testset "Different means, same workspace" begin
        ws = GMRFWorkspace(Q)
        μ_a = zeros(n)
        μ_b = ones(n)

        wg_a = WorkspaceGMRF(μ_a, Q, ws)
        wg_b = WorkspaceGMRF(μ_b, Q, ws)

        z = randn(n)
        @test mean(wg_a) == μ_a
        @test mean(wg_b) == μ_b
        @test logpdf(wg_a, z) ≈ logpdf(GMRF(μ_a, Q), z) rtol = 1.0e-10
        @test logpdf(wg_b, z) ≈ logpdf(GMRF(μ_b, Q), z) rtol = 1.0e-10
    end

    @testset "logpdf(prior) consistent across gaussian_approximation" begin
        # GA leaves ws factorized at Q_post; a subsequent logpdf(prior) must
        # still evaluate against Q_prior (both quadratic form and logdet).
        using Distributions: Poisson
        Q_prior = spdiagm(-1 => fill(-0.3, 4), 0 => fill(2.0, 5), 1 => fill(-0.3, 4))
        y_obs = [2, 1, 3, 0, 4]
        z_eval = randn(5); z_eval .-= sum(z_eval) / 5
        ws = GMRFWorkspace(copy(Q_prior))
        prior = WorkspaceGMRF(zeros(5), copy(Q_prior), ws)

        lp_ref = logpdf(GMRF(zeros(5), copy(Q_prior)), z_eval)
        lp_before = logpdf(prior, z_eval)
        @test lp_before ≈ lp_ref rtol = 1.0e-10

        obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y_obs))
        _ = gaussian_approximation(prior, obs_lik)

        lp_after = logpdf(prior, z_eval)
        @test lp_after ≈ lp_ref rtol = 1.0e-10
    end

    @testset "gaussian_approximation reseeds prior_nzval from owner" begin
        # When two priors share a workspace, GA on one prior must seed its
        # Newton loop from that prior's snapshot, not from whichever prior
        # last touched ws.
        using Distributions: Poisson
        Q_a = spdiagm(0 => fill(2.0, 5), 1 => fill(-0.3, 4), -1 => fill(-0.3, 4))
        Q_b = spdiagm(0 => fill(5.0, 5), 1 => fill(-0.7, 4), -1 => fill(-0.7, 4))
        y = [2, 1, 3, 0, 4]
        x = zeros(5)
        ws = GMRFWorkspace(copy(Q_a))
        wg_a = WorkspaceGMRF(zeros(5), copy(Q_a), ws)
        wg_b = WorkspaceGMRF(zeros(5), copy(Q_b), ws)

        # Touch wg_b last so ws holds Q_b's data.
        _ = logpdf(wg_b, randn(5))
        obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))

        post_ws = gaussian_approximation(wg_a, obs_lik)
        post_ref = gaussian_approximation(GMRF(zeros(5), copy(Q_a)), obs_lik)

        @test mean(post_ws) ≈ mean(post_ref) rtol = 1.0e-8
        @test logpdf(post_ws, x) ≈ logpdf(post_ref, x) rtol = 1.0e-8
    end
end
