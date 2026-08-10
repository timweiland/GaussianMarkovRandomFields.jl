using LinearAlgebra
using SparseArrays
using LinearMaps
using CliqueTrees.Multifrontal: ChordalCholesky

export gaussian_approximation

# Sparse-preserving subtraction for Hermitian matrices
hermdiff(A::Hermitian, B) = Hermitian(parent(A) - B, Symbol(A.uplo))

function neg_log_posterior(prior_gmrf::AbstractGMRF, obs_lik::ObservationLikelihood, x)
    return -logpdf(prior_gmrf, x) - loglik(x, obs_lik)
end

function ∇ₓ_neg_log_posterior(prior_gmrf::AbstractGMRF, obs_lik::ObservationLikelihood, x)
    return -gradlogpdf(prior_gmrf, x) - loggrad(x, obs_lik)
end

function ∇²ₓ_neg_log_posterior(prior_gmrf::AbstractGMRF, obs_lik::ObservationLikelihood, x)
    return precision_matrix(prior_gmrf) - loghessian(x, obs_lik)
end

# Private dispatch: extract constraints if present
_extract_constraints(::GMRF) = nothing
_extract_constraints(constrained::ConstrainedGMRF) = (A = constrained.constraint_matrix, e = constrained.constraint_vector)
_extract_constraints(::ChordalGMRF) = nothing

# Private dispatch: apply constraints to result
_apply_constraints(gmrf::GMRF, ::Nothing) = gmrf
_apply_constraints(gmrf::GMRF, constraints::NamedTuple) = ConstrainedGMRF(gmrf, constraints.A, constraints.e)

# Private dispatch: extract base GMRF for optimization
_base_gmrf(gmrf::GMRF) = gmrf
_base_gmrf(constrained::ConstrainedGMRF) = constrained.base_gmrf
_base_gmrf(gmrf::ChordalGMRF) = gmrf

# Compute the constrained Newton step via the KKT Schur complement.
# Solves H⁻¹Aᵀ using the existing linsolve cache (m sparse solves, m = #constraints),
# then removes the constraint-normal component so AΔx = 0.
_constrain_step(step, cache, ::Nothing) = step
function _constrain_step(step, cache, constraints::NamedTuple)
    A = constraints.A
    m = size(A, 1)
    n = length(step)

    # Solve H · X = Aᵀ column-by-column (reuses current cache factorization)
    A_tilde_T = Matrix{eltype(step)}(undef, n, m)
    saved_b = copy(cache.b)
    for i in 1:m
        cache.b .= @view(A[i, :])
        A_tilde_T[:, i] .= solve!(cache).u
    end
    cache.b .= saved_b

    # Schur complement: remove constraint-normal component
    L_c = cholesky(Symmetric(A * A_tilde_T))
    return step - A_tilde_T * (L_c \ (A * step))
end

# Set the matrix in the linsolve cache to Q
_update_linsolve_cache!(cache, Q) = _update_linsolve_cache_inner!(cache, Q, cache.alg)

function _update_linsolve_cache_inner!(cache, Q, alg)
    return cache.A = Q
end

function _update_linsolve_cache_inner!(cache, Q, alg::LinearSolve.DefaultLinearSolver)
    actual_alg = LinearSolve.algchoice_to_alg(Symbol(alg.alg))
    return _update_linsolve_cache_inner!(cache, Q, actual_alg)
end

function _update_linsolve_cache_inner!(cache, Q, alg::LinearSolve.LDLtFactorization)
    # LDLtFactorization mutates the original matrix, so we need to copy it
    # Causes super nasty and sneaky bugs if we don't do this!
    return cache.A = copy(Q)
end

# Solver abstraction for the gaussian_approximation Newton iteration.
# A "solver" is either a `LinearSolve` cache (CHOLMOD-backed GMRF /
# ConstrainedGMRF / non-Gaussian latent priors) or a `ChordalCholesky`
# (ChordalGMRF). The Newton loop is generic over it: `_ga_refactor!`
# updates the factorization to the posterior precision `Q_prior - H`, and
# `_ga_solve` solves against it. Splitting refactor from solve lets the
# final `_build_posterior` refresh the factorization without a wasted
# triangular solve.
_ga_init_solver(gmrf::ChordalGMRF, _prior, _obs_lik, _x_init) = copy(gmrf.F)

function _ga_init_solver(gmrf::GMRF, prior, obs_lik::ObservationLikelihood, x_init)
    Q_p, = _prior_local(prior, x_init)
    return _ga_resolve_cache(linsolve_cache(gmrf), Q_p - loghessian(x_init, obs_lik))
end

# The Newton loop factorizes `Q_prior - H`, whose storage type need not match the prior
# precision the cache was built for: a `SymTridiagonal` AR(1)/RW(1) prior against a sparse
# observation Hessian gives a sparse posterior precision, and the prior's
# `LDLtFactorization` cannot factorize that (`ldlt!` only accepts `SymTridiagonal`).
#
# Reuse the prior's cache whenever the storage type carries over — that keeps the fast path
# for a `SymTridiagonal` prior meeting a `Diagonal` Hessian, which stays `SymTridiagonal`.
# Otherwise build a fresh cache with an algorithm resolved for the posterior precision.
# Resolving once here rather than per iteration keeps `solver` concretely typed inside
# `_newton_loop`; this call is a function barrier, so the union return costs one dispatch.
function _ga_resolve_cache(cache::LinearSolve.LinearCache, Q_posterior)
    if typeof(prepare_for_linsolve(Q_posterior, cache.alg)) === typeof(cache.A)
        return deepcopy(cache)
    end
    A_linsolve, resolved_alg = resolve_linsolve(Q_posterior, cache.alg)
    return init(LinearProblem(A_linsolve, copy(cache.b)), resolved_alg)
end

function _ga_refactor!(cache::LinearSolve.LinearCache, Q_prior, H)
    Q_new = prepare_for_linsolve(Q_prior - H, cache.alg)
    _update_linsolve_cache!(cache, Q_new)
    return Q_new
end

function _ga_refactor!(F::ChordalCholesky, Q_prior, H)
    Q_new = hermdiff(Q_prior, H)
    copyto!(F, Q_new)
    cholesky!(F)
    return Q_new
end

_ga_solve(cache::LinearSolve.LinearCache, b) = (cache.b = b; copy(solve!(cache).u))
_ga_solve(F::ChordalCholesky, b) = F \ b

function _ga_make_posterior(x, Q, solver, prior::Union{GMRF, ConstrainedGMRF, LatentPrior}, constraints)
    new_gmrf = GMRF(x, Q; linsolve_cache = solver)
    return _apply_constraints(new_gmrf, constraints)
end

function _ga_make_posterior(x, Q, solver, prior::ChordalGMRF, _)
    # ChordalGMRF priors do not carry constraints (see _extract_constraints(::ChordalGMRF) = nothing),
    # so the second arg is always `nothing` at runtime. The relaxed signature is just so JET
    # can see a method for the inferred Union type of the constraints argument.
    return ChordalGMRF(x, Q, solver)
end

"""
    gaussian_approximation(prior_gmrf, obs_lik; kwargs...) -> AbstractGMRF

Find Gaussian approximation to the posterior using Fisher scoring.

This function finds the mode of the posterior distribution and constructs a Gaussian
approximation around it using Fisher scoring (Newton-Raphson with Fisher information matrix).

Works for `GMRF`, `ConstrainedGMRF`, and `ChordalGMRF` priors, automatically handling
constraint projection when needed.

# Arguments
- `prior_gmrf`: Prior GMRF distribution for the latent field (GMRF, ConstrainedGMRF, or ChordalGMRF)
- `obs_lik`: Materialized observation likelihood (contains data and hyperparameters)

# Keyword Arguments
- `max_iter::Int=50`: Maximum number of Fisher scoring iterations
- `mean_change_tol::Real=1e-4`: Convergence tolerance for mean change
- `newton_dec_tol::Real=1e-5`: Newton decrement convergence tolerance
- `adaptive_stepsize::Bool=true`: Enable adaptive stepsize with backtracking line search
- `max_linesearch_iter::Int=10`: Maximum line search iterations per Newton step
- `step_recovery::Symbol=:retry_full`: How the line search recovers the step scale `α`
  after a backtrack. `:retry_full` probes the undamped step (`α = 1`) at the start of
  every Newton iteration and only falls back to the damped scale if it is rejected;
  `:sqrt` instead grows `α` as `sqrt(α)` per accepted step, spreading the recovery over
  ~8–10 damped (and therefore fully factorized) Newton iterations. Use `:sqrt` if the
  merit is not convex along the Newton direction and the slow recovery is wanted as an
  oscillation guard.
- `predictive_convergence::Bool=true`: When quadratic contraction predicts that the next
  Newton decrement clears `newton_dec_tol`, finish against the factorization already in
  hand instead of refactorizing for the final step. The step's own decrement is the
  convergence certificate, and a couple of further steps on the same factorization
  polish the mode — a chord iteration, which converges to the true mode using solves
  rather than a refactorization. Set to `false` to refactorize for the final step
  instead.
- `verbose::Bool=false`: Print iteration information

# Returns
- Gaussian approximation to the posterior p(x | θ, y) (same type as input prior)

# Example
```julia
# Set up components
prior_gmrf = GMRF(μ_prior, Q_prior)
obs_model = ExponentialFamily(Poisson)
obs_lik = obs_model(y)

# Find Gaussian approximation - uses adaptive stepsize by default
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)

# For well-conditioned problems, disable adaptive stepsize for speed
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik; adaptive_stepsize=false)
```
"""
function gaussian_approximation(
        prior_gmrf::Union{GMRF, ConstrainedGMRF, ChordalGMRF},
        obs_lik::ObservationLikelihood;
        x0::Union{Nothing, AbstractVector} = nothing,
        max_iter::Int = 50,
        mean_change_tol::Real = 1.0e-4,
        newton_dec_tol::Real = 1.0e-5,
        adaptive_stepsize::Bool = true,
        max_linesearch_iter::Int = 10,
        step_recovery::Symbol = :retry_full,
        predictive_convergence::Bool = true,
        verbose::Bool = false
    )
    base_gmrf = _base_gmrf(prior_gmrf)
    constraints = _extract_constraints(prior_gmrf)
    x_init = x0 === nothing ? mean(prior_gmrf) : x0
    solver = _ga_init_solver(base_gmrf, prior_gmrf, obs_lik, x_init)
    return _newton_loop(
        prior_gmrf, obs_lik, solver, constraints, x_init;
        max_iter, mean_change_tol, newton_dec_tol,
        adaptive_stepsize, max_linesearch_iter, step_recovery,
        predictive_convergence, verbose,
    )
end

# Resolve the `step_recovery` keyword to "probe the full Newton step first?".
# Validated here — the two Newton loops are the single funnel every
# `gaussian_approximation` entry point passes through.
function _retry_full_step(step_recovery::Symbol)
    step_recovery === :retry_full && return true
    step_recovery === :sqrt && return false
    throw(
        ArgumentError(
            "step_recovery must be :retry_full or :sqrt, got :$step_recovery"
        )
    )
end

# Line-search merit: the neg-log-posterior up to an x-independent constant.
_ga_merit(prior, Q_p, h, obs_lik, x) = _prior_energy(prior, Q_p, h, x) - loglik(x, obs_lik)

# Rounding-noise floor of a merit evaluation. The merit is a difference of two
# O(n)-term sums (prior energy and log-likelihood), so its achievable precision
# scales with the magnitude of *those* terms — `abs(merit)` alone underestimates it
# whenever the two nearly cancel. A few dozen ULPs: the artifacts measured on the
# convergence plateau are 1-6 ULPs, while the smallest decrease the line search can
# still act on there is ~1e-11 relative, so this sits well clear of both.
#
# `eps(typeof(scale))` rather than `eps(scale)`: the `ChordalGMRF` hyperparameter path
# differentiates straight through the Newton loop, so `scale` can be a `ForwardDiff.Dual`
# (for which the type-based `eps` is the value type's). The tolerance is then a Dual too,
# which is harmless — `obj_accept` is only ever compared with `<=`, on values alone.
_merit_atol(scale::Real) = 64 * eps(typeof(scale)) * max(scale, one(scale))

# Backtracking line search shared by the cache- and workspace-backed Newton loops.
# Returns the accepted iterate `x_new` and the (possibly shrunk) step scale `α`. The
# merit is `_prior_energy(prior, Q_p, h, ·) - loglik(·, obs_lik)` — the neg-log-posterior
# up to an x-independent constant, so accept/reject decisions match the full merit while
# never evaluating `logpdf` (no factorization on a shared workspace).
#
# `α` persists across Newton iterations: a backtrack shrinks it 10×, an accepted step
# grows it back as `sqrt(α)`. With `retry_full` (the `:retry_full` policy) the undamped
# step is probed before that ladder is descended, which is what keeps a single early
# backtrack from spreading its recovery over ~8-10 damped Newton iterations. The probe is
# cheap precisely because the merit never factorizes, whereas a damped iteration costs a
# full factorize + solve.
function _ga_line_search(
        prior, Q_p, h, energy_k, obs_lik, x_k, step, α;
        max_linesearch_iter::Int, newton_dec_tol::Real, verbose::Bool,
        retry_full::Bool,
    )
    loglik_k = loglik(x_k, obs_lik)
    obj_current = energy_k - loglik_k
    # Accept anything that is not a *significant* increase. Near the mode the true
    # decrease falls below the merit's own rounding noise, and a strict `<=` then
    # rejects a perfectly good undamped Newton step over a 1-2 ULP artifact and
    # replaces it with a 10x-damped one — which is how a warm start from a converged
    # mode came back with ‖∇ₓ neg-log-posterior‖∞ ≈ 5e-10 instead of ≈ 8e-15. A step
    # whose merit change is at the noise floor cannot be an overshoot, so there is
    # nothing for the line search to guard against.
    obj_accept = obj_current + _merit_atol(abs(energy_k) + abs(loglik_k))

    if retry_full && α < one(α)
        x_full = x_k - step
        if _ga_merit(prior, Q_p, h, obs_lik, x_full) <= obj_accept
            verbose && println("    Accepted full step (α=1)")
            return x_full, one(α)
        end
        verbose && println("    Full step rejected, resuming at α=$(round(α, digits = 4))")
    end

    accept = false
    # Pre-initialize so `x_new` is provably defined for static analysis: the loop assigns
    # it only on the accept branch, in a way JET cannot track. Always overwritten below —
    # by the loop on acceptance, or by the `!accept` fallback.
    x_new = x_k - α * step

    for ls_iter in 1:max_linesearch_iter
        candidate = x_k - α * step
        obj_candidate = _ga_merit(prior, Q_p, h, obs_lik, candidate)

        if obj_candidate <= obj_accept
            x_new = candidate
            α = sqrt(α)
            accept = true
            verbose && ls_iter > 1 && println("    Accepted at α=$(round(α^2, digits = 3)) after $ls_iter backtracks")
            break
        else
            α *= 0.1
            verbose && println("    Backtrack: α=$(round(α, digits = 4))")
            if α * norm(step, Inf) < newton_dec_tol / 1000
                x_new = candidate
                accept = true
                break
            end
        end
    end

    accept || (x_new = x_k - α * step)
    return x_new, α
end

# Quadratic-convergence look-ahead, shared by both Newton loops.
#
# The convergence test measures the decrement at `x_k` and then returns
# `x_new = x_k - step`, so a solve that is already in the quadratic regime
# spends a whole iteration purely to certify a decrement the previous two
# iterations already imply. That iteration is dominated by its refactorization
# (88% of iteration cost on a 1D n=500 fit, ~95% on 2D/3D grids), and the
# Hessian it rebuilds differs from the one already factored by `O(‖Δx‖)` — a
# perturbation invisible next to `newton_dec_tol`.
#
# So instead of skipping the confirming iteration, run it against the
# factorization already in hand: evaluate the gradient at `x_new`, solve with
# the stale factor, and use the decrement of that step as the certificate.
# `_frozen_finish` then keeps stepping on the same factorization, which lands
# the mode at least as close to the true one as the refactorized iteration
# would have — the refactorization is simply not paid.
#
# `_predict_converged` is only the gate that decides when to attempt it. For
# undamped Newton steps the decrement contracts quadratically,
# `dec_{k+1} ≈ C²·dec_k²`, and `C²` is observable from the last two decrements
# as `dec_k / dec_{k-1}²`, giving `dec_pred = dec_k · (dec_k / dec_{k-1})²`.
# The prediction needs no safety margin here — it is checked, not trusted: a
# wrong gate costs one extra solve (~12% of an iteration) and the loop carries
# on. Guards:
#
# - `iter > 1`, so `dec_prev` is a real decrement. It is then also strictly
#   positive: a non-positive decrement would have converged at iteration
#   `iter - 1`.
# - `α == 1 && α_prev == 1`: this step and the previous one were both undamped,
#   so `dec_k` and `dec_prev` are two full Newton steps and their ratio means
#   what the formula assumes. The two scales are checked separately because
#   `α == 1` alone no longer implies the previous step was undamped: under
#   `step_recovery = :retry_full` the line search returns exactly `1.0` the
#   first time it accepts the full step after a backtrack, whereas the `:sqrt`
#   policy could only creep back to `1.0` over many undamped iterations. While
#   the line search is still damping, the quadratic reasoning does not apply and
#   the attempt would mostly be wasted.
function _predict_converged(dec_k, dec_prev, α, α_prev, newton_dec_tol, iter)
    (iter > 1 && α == 1.0 && α_prev == 1.0) || return false
    return dec_k * (dec_k / dec_prev)^2 < newton_dec_tol
end

# Gradient of the neg-log-posterior at `x`, in the natural form both Newton
# loops build it: the local quadratic at `x` has `∇log p_prior(x) = h - Q·x`.
function _ga_neg_score(prior, obs_lik::ObservationLikelihood, x)
    Q_p, h = _prior_local(prior, x)
    return (Q_p * x - h) .- loggrad(x, obs_lik)
end

# Newton steps taken against the frozen factorization once the look-ahead fires.
# The gradient is exact at every step and only the Hessian is stale, so this is a
# chord iteration on the true stationarity condition `∇f(x) = 0`: it converges to
# the true mode, linearly, at a rate set by how far the Hessian has moved.
#
# Only the *first* step decides anything — its decrement is the convergence
# certificate, the same quantity the loop's own test uses. The rest cannot change
# that decision; they only walk the mode closer to the true one, so this count is
# a quality/cost dial and not a hidden correctness threshold. Its floor is "the
# returned mode meets `newton_dec_tol`", which step one already establishes.
#
# Three is a budget, bounded well under what it buys: three solves is ~1/3 of the
# refactorization being skipped on a 1D n=500 fit and ~1/10 on 2D/3D grids. The
# per-step contraction ranges from ~20x (small, ill-scaled fits where the
# look-ahead fires while the Hessian is still moving) to ~1e6x (warm solves deep
# in the quadratic regime), and across measured Poisson, binomial,
# negative-binomial and custom-likelihood fits three steps land the mode at or
# inside where a refactorized final iteration would have put it. Adaptive
# stopping rules were measured too — on decrement depth, or on the step-to-step
# improvement stalling — and neither beat a flat count: the depth rules stop
# short exactly where the polish matters, the stall rules spend more steps for no
# gain.
const _FROZEN_FINISH_STEPS = 3

# `solve_step` maps a gradient to the (constraint-projected) Newton step against the
# factorization the caller already holds — nothing in here invalidates it. Returns the
# accepted iterate, or `nothing` when the look-ahead was wrong and the loop should
# continue with a refactorization.
function _frozen_finish(
        prior, obs_lik::ObservationLikelihood, solve_step::F, x,
        newton_dec_tol::Real, verbose::Bool, iter::Int,
    ) where {F}
    for j in 1:_FROZEN_FINISH_STEPS
        g = _ga_neg_score(prior, obs_lik, x)
        step = solve_step(g)
        dec = dot(g, step)
        if j == 1
            dec < newton_dec_tol || return nothing
            verbose && println(
                "  Iter $(iter + 1): Newton dec = $(round(dec, sigdigits = 3)) (reused factorization)"
            )
        end
        x = x - step
    end
    return x
end

"""
    _newton_loop(prior, obs_lik, solver, constraints, x_init; ...) -> AbstractGMRF

Shared Newton loop for `gaussian_approximation`. Two abstractions compose
here:

- The prior side is queried per iterate via `_prior_local(prior, x_k) ->
  (Q, h, energy)`. For Gaussian priors `(Q, h)` are constant in `x_k`;
  for the non-Gaussian `LatentPrior` adapter they re-linearise. The
  line-search merit never calls `logpdf`, so a shared `GMRFWorkspace` is
  not refactorized at the prior precision.
- The linear-algebra side is the `solver`: a `LinearSolve` cache
  (CHOLMOD) or a `ChordalCholesky` (`ChordalGMRF`). `_ga_refactor!`
  updates it to `Q_prior - H`; `_ga_solve` solves the Newton system.

`constraints` is `nothing` or `(A=..., e=...)` for KKT step projection.
"""
function _newton_loop(
        prior, obs_lik::ObservationLikelihood,
        solver, constraints, x_init::AbstractVector;
        max_iter::Int,
        mean_change_tol::Real,
        newton_dec_tol::Real,
        adaptive_stepsize::Bool,
        max_linesearch_iter::Int,
        step_recovery::Symbol,
        predictive_convergence::Bool,
        verbose::Bool,
    )
    retry_full = _retry_full_step(step_recovery)
    x_k = copy(x_init)
    α = 1.0
    dec_prev = 0.0
    verbose && println("Starting Fisher scoring...")

    for iter in 1:max_iter
        Q_p, h, energy_k = _prior_local(prior, x_k)
        H_k = loghessian(x_k, obs_lik)
        g_l = loggrad(x_k, obs_lik)

        _ga_refactor!(solver, Q_p, H_k)
        # ∇ₓ neg_log_posterior(x_k) = -∇log p_prior(x_k) - ∇log p_lik(x_k);
        # prior gradient in natural form is ∇log p(x_k) = h - Q_p · x_k.
        neg_score_k = (Q_p * x_k - h) .- g_l
        step = _ga_solve(solver, neg_score_k)
        step = _constrain_step(step, solver, constraints)

        # The scale the previous iteration accepted, kept before the line search
        # overwrites `α`: `_predict_converged` needs both to be undamped.
        α_prev = α
        if adaptive_stepsize
            x_new, α = _ga_line_search(
                prior, Q_p, h, energy_k, obs_lik, x_k, step, α;
                max_linesearch_iter, newton_dec_tol, verbose, retry_full,
            )
        else
            x_new = x_k - step
        end

        newton_decrement = dot(neg_score_k, step)
        mean_change = norm(x_new - x_k)
        mean_change_rel = mean_change / max(norm(x_k), 1.0e-10)
        verbose && println("  Iter $iter: Newton dec = $(round(newton_decrement, sigdigits = 3)), α = $(round(α, digits = 3))")

        if (newton_decrement < newton_dec_tol) || (mean_change < mean_change_tol) || (mean_change_rel < mean_change_tol)
            verbose && println("  Converged after $iter iterations")
            return _build_posterior(prior, obs_lik, solver, constraints, x_new)
        end

        if predictive_convergence &&
                _predict_converged(newton_decrement, dec_prev, α, α_prev, newton_dec_tol, iter)
            x_final = _frozen_finish(
                prior, obs_lik, g -> _constrain_step(_ga_solve(solver, g), solver, constraints),
                x_new, newton_dec_tol, verbose, iter,
            )
            if x_final !== nothing
                verbose && println("  Converged after $(iter + 1) iterations")
                return _build_posterior(prior, obs_lik, solver, constraints, x_final)
            end
        end

        dec_prev = newton_decrement
        x_k = x_new
    end

    verbose && println("  Reached max_iter without convergence")
    return _build_posterior(prior, obs_lik, solver, constraints, x_k)
end

# Refresh the factorization at the final iterate so the posterior
# `(mean, precision)` are consistent (the loop's last `_ga_refactor!`
# happened at `x_k`, not `x_new`). Matters for AD gradients on
# hyperparameter objectives where small drift turns into ~1e-4 error.
function _build_posterior(prior, obs_lik, solver, constraints, x_final)
    Q_p, = _prior_local(prior, x_final)
    H_final = loghessian(x_final, obs_lik)
    Q_final = _ga_refactor!(solver, Q_p, H_final)
    return _ga_make_posterior(x_final, Q_final, solver, prior, constraints)
end

# Specialized dispatch for Normal observation likelihoods with identity link (conjugate prior case)
function gaussian_approximation(prior_gmrf::GMRF, obs_lik::NormalLikelihood{IdentityLink})
    # Normal observations with identity link: y ~ N(x, σ²I) - this is conjugate!
    # Equivalent to: y = A*x + 0 + ε, where ε ~ N(0, σ²I)

    Q_ϵ = obs_lik.inv_σ²  # 1/σ² (scalar gets converted to scaled identity automatically)
    y = obs_lik.y
    b = zeros(length(y))  # No offset

    if obs_lik.indices === nothing
        # Non-indexed case: A = I, so A' * Q_ϵ * A = Q_ϵ * I = Diagonal(fill(Q_ϵ, n))
        A = 1.0 * I
        n_total = length(mean(prior_gmrf))
        obs_precision_contrib = Diagonal(fill(obs_lik.inv_σ², n_total))
        return linear_condition(
            prior_gmrf; A = A, Q_ϵ = Q_ϵ, y = y, b = b,
            obs_precision_contrib = obs_precision_contrib
        )
    else
        # Indexed case: A' * Q_ϵ * A is diagonal with Q_ϵ at selected indices, 0 elsewhere
        n_total = length(mean(prior_gmrf))
        n_obs = length(obs_lik.y)
        A = spzeros(n_obs, n_total)
        diag_entries = zeros(n_total)
        @inbounds for (i, idx) in enumerate(obs_lik.indices)
            A[i, idx] = 1.0
            diag_entries[idx] = obs_lik.inv_σ²
        end
        return linear_condition(
            prior_gmrf; A = A, Q_ϵ = Q_ϵ, y = y, b = b,
            obs_precision_contrib = Diagonal(diag_entries)
        )
    end
end

# Affine offset of a LinearlyTransformedLikelihood as a concrete vector for the
# conjugate path: `y ~ N(A·x + b, σ²)` ⇒ `y = A·x + b + ε`, matching linear_condition's `b`.
_offset_or_zeros(::Nothing, n::Integer) = zeros(n)
_offset_or_zeros(b::AbstractVector, ::Integer) = b

# Specialized dispatch for linearly transformed Normal observation likelihoods (also conjugate)
function gaussian_approximation(prior_gmrf::GMRF, obs_lik::LinearlyTransformedLikelihood{<:NormalLikelihood{IdentityLink}})
    # Linearly transformed Normal with identity link: y ~ N(A*x + b, σ²I) - still conjugate!
    # This is exactly the linear conditioning setup: y = A*x + b + ε, where ε ~ N(0, σ²I)

    base_lik = obs_lik.base_likelihood
    A = obs_lik.design_matrix
    Q_ϵ = base_lik.inv_σ²  # 1/σ² (scalar gets converted to scaled identity automatically)
    y = base_lik.y
    b = _offset_or_zeros(obs_lik.offset, length(y))

    return linear_condition(prior_gmrf; A = A, Q_ϵ = Q_ϵ, y = y, b = b)
end

# Specialized dispatch for ConstrainedGMRF with Normal observations (conjugate case)
function gaussian_approximation(prior_constrained::ConstrainedGMRF, obs_lik::NormalLikelihood{IdentityLink})
    # Delegate to linear_condition for conjugate case
    if obs_lik.indices === nothing
        A = I
    else
        A = _build_index_matrix(obs_lik.indices, length(prior_constrained))
    end

    return linear_condition(
        prior_constrained;
        A = A,
        Q_ϵ = obs_lik.inv_σ²,
        y = obs_lik.y,
        b = zeros(length(obs_lik.y))
    )
end

# Helper function to build index matrix for indexed observations
function _build_index_matrix(indices, n_total)
    n_obs = length(indices)
    A = spzeros(n_obs, n_total)
    for (i, idx) in enumerate(indices)
        A[i, idx] = 1.0
    end
    return A
end

# Specialized dispatch for ConstrainedGMRF with linearly transformed Normal observations (conjugate case)
function gaussian_approximation(prior_constrained::ConstrainedGMRF, obs_lik::LinearlyTransformedLikelihood{<:NormalLikelihood{IdentityLink}})
    # Delegate to linear_condition for conjugate case
    base_lik = obs_lik.base_likelihood
    return linear_condition(
        prior_constrained;
        A = obs_lik.design_matrix,
        Q_ϵ = base_lik.inv_σ²,
        y = base_lik.y,
        b = _offset_or_zeros(obs_lik.offset, length(base_lik.y))
    )
end

# MetaGMRF dispatches - preserve wrapper type and metadata
function gaussian_approximation(prior_mgmrf::MetaGMRF, obs_lik::ObservationLikelihood; kwargs...)
    posterior_gmrf = gaussian_approximation(prior_mgmrf.gmrf, obs_lik; kwargs...)
    return MetaGMRF(posterior_gmrf, prior_mgmrf.metadata)
end
