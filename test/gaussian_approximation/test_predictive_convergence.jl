using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Distributions

# Predictive convergence (issue #202): when quadratic contraction predicts that the
# next Newton decrement clears `newton_dec_tol`, the solve finishes against the
# factorization already in hand instead of refactorizing for the final step. The first
# step's decrement is the convergence certificate; the remaining steps are a chord
# iteration on the same factorization, which converges to the true mode.

# True Newton decrement gᵀH⁻¹g at the returned mode. `_build_posterior` refreshes the
# factorization at the returned iterate, so the posterior precision *is* H there.
function _returned_decrement(prior, obs_lik, posterior)
    x = mean(posterior)
    g = GaussianMarkovRandomFields.∇ₓ_neg_log_posterior(prior, obs_lik, x)
    return dot(g, cholesky(Symmetric(sparse(precision_matrix(posterior)))) \ g)
end

# Number of Newton loop trips a run took, recovered without touching internals:
# truncating `max_iter` below the convergence iteration changes the returned mode,
# and at or above it the run is bit-identical.
function _newton_iterations(run, converged_mean; max_probe = 30)
    for m in 1:max_probe
        mean(run(m)) == converged_mean && return m
    end
    return -1
end

@testset "Predictive convergence exit" begin

    @testset "look-ahead gate" begin
        pc = GaussianMarkovRandomFields._predict_converged
        tol = 1.0e-8

        # Quadratic contraction, undamped: the next decrement is predicted to clear tol.
        @test pc(1.0e-6, 1.0e-2, 1.0, tol, 2)

        # First iteration: no previous decrement to estimate contraction from.
        @test !pc(1.0e-6, 1.0e-2, 1.0, tol, 1)

        # Damped step (α < 1): quadratic reasoning does not apply.
        @test !pc(1.0e-6, 1.0e-2, 0.999, tol, 2)

        # Contraction too slow for the next step to clear tol.
        @test !pc(1.0e-4, 1.0e-2, 1.0, tol, 2)

        # No contraction at all.
        @test !pc(1.0e-2, 1.0e-2, 1.0, tol, 2)
    end

    # Warm solve in the pattern of an outer hyperparameter loop: solve once, then
    # re-solve a nearby model starting from the first mode.
    n = 200
    Q = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.01, n), 1 => fill(-1.0, n - 1))
    x_true = 2 .* sin.(range(0, 4π; length = n))
    y = PoissonObservations(round.(Int, exp.(x_true)))
    obs_lik = ExponentialFamily(Poisson)(y)
    warm_start = mean(gaussian_approximation(GMRF(zeros(n), Q), obs_lik))
    prior = GMRF(zeros(n), 1.1 * Q)
    # mean_change pinned tiny so the decrement criterion is the binding one.
    tolkw = (newton_dec_tol = 1.0e-8, mean_change_tol = 1.0e-12)

    @testset "final steps reuse the factorization" begin
        post_on = gaussian_approximation(prior, obs_lik; x0 = warm_start, tolkw...)
        post_off = gaussian_approximation(
            prior, obs_lik; x0 = warm_start, predictive_convergence = false, tolkw...
        )

        # One fewer trip through the loop: the last Newton step rides along inside the
        # previous iteration instead of getting one of its own.
        iters_on = _newton_iterations(
            m -> gaussian_approximation(prior, obs_lik; x0 = warm_start, max_iter = m, tolkw...),
            mean(post_on),
        )
        iters_off = _newton_iterations(
            m -> gaussian_approximation(
                prior, obs_lik; x0 = warm_start, max_iter = m,
                predictive_convergence = false, tolkw...
            ),
            mean(post_off),
        )
        @test iters_off > 1
        @test iters_on == iters_off - 1

        # Same mode as the refactorizing path, and the chord iteration lands it far
        # inside the tolerance rather than merely under it.
        @test mean(post_on) ≈ mean(post_off) rtol = 1.0e-8
        @test precision_matrix(post_on) ≈ precision_matrix(post_off) rtol = 1.0e-8
        @test _returned_decrement(prior, obs_lik, post_on) < 1.0e-3 * tolkw.newton_dec_tol
        @test _returned_decrement(prior, obs_lik, post_off) < tolkw.newton_dec_tol
    end

    @testset "workspace path" begin
        ws = GMRFWorkspace(sparse(1.1 * Q))
        ws_prior = WorkspaceGMRF(zeros(n), sparse(1.1 * Q), ws)
        post_on = gaussian_approximation(ws_prior, obs_lik; x0 = warm_start, tolkw...)
        post_off = gaussian_approximation(
            ws_prior, obs_lik; x0 = warm_start, predictive_convergence = false, tolkw...
        )

        iters_on = _newton_iterations(
            m -> gaussian_approximation(ws_prior, obs_lik; x0 = warm_start, max_iter = m, tolkw...),
            mean(post_on),
        )
        iters_off = _newton_iterations(
            m -> gaussian_approximation(
                ws_prior, obs_lik; x0 = warm_start, max_iter = m,
                predictive_convergence = false, tolkw...
            ),
            mean(post_off),
        )
        @test iters_on == iters_off - 1
        @test mean(post_on) ≈ mean(post_off) rtol = 1.0e-6

        # Same mode as the cache-backed path.
        cache_post = gaussian_approximation(prior, obs_lik; x0 = warm_start, tolkw...)
        @test mean(post_on) ≈ mean(cache_post) rtol = 1.0e-10
    end

    @testset "no effect while the line search is damping" begin
        # Extreme counts under a weak prior: the line search backtracks on the first
        # step and α only creeps back towards 1, so the gate never opens and the
        # solve is bit-identical with and without the look-ahead.
        m = 5
        weak_prior = GMRF(zeros(m), 0.01 * sparse(I, m, m))
        extreme_lik = ExponentialFamily(Poisson)(PoissonObservations([200, 50, 500, 10, 1000]))
        post_on = gaussian_approximation(weak_prior, extreme_lik; tolkw...)
        post_off = gaussian_approximation(
            weak_prior, extreme_lik; predictive_convergence = false, tolkw...
        )
        @test mean(post_on) == mean(post_off)
    end

    @testset "latent-model entry point threads the keyword" begin
        model = RW1Model(n)
        post_on = gaussian_approximation(model, obs_lik; τ = 1.0, x0 = warm_start, tolkw...)
        post_off = gaussian_approximation(
            model, obs_lik; τ = 1.0, x0 = warm_start,
            predictive_convergence = false, tolkw...
        )
        @test mean(post_on) ≈ mean(post_off) rtol = 1.0e-6
    end
end
