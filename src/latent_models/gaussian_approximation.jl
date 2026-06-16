export marginal_loglikelihood

"""
    marginal_loglikelihood(prior::AbstractLatentPrior, obs_lik, posterior::AbstractGMRF; θ...) -> Real

Laplace approximation to `log p(y | θ) = log ∫ p(x | θ) p(y | x, θ) dx`,
using the posterior mode and precision returned by
`gaussian_approximation`. Computed via the identity

    log p(y | θ) ≈ log p(x* | θ) + log p(y | x*, θ) - log p_Laplace(x* | y, θ)

evaluated at the converged mode `x*`. The posterior's `logpdf` carries
the constraint correction for `ConstrainedGMRF`-style posteriors, so the
formula handles unconstrained and constrained cases uniformly.
"""
function marginal_loglikelihood(
        prior::AbstractLatentPrior, obs_lik::ObservationLikelihood, posterior::AbstractGMRF;
        kwargs...,
    )
    θ = NamedTuple(kwargs)
    x_star = mean(posterior)
    # Use the scalar `prior_logdensity` hook rather than `local_quadratic(...).logp_ref`:
    # it avoids rebuilding the (sparse) Hessian just to read a number, and — for an
    # AutoDiffLatentPrior under an outer ForwardDiff θ-pass — it evaluates `log p` as
    # plain arithmetic, so it doesn't trip the sparsity tracer on Dual inputs.
    return prior_logdensity(prior, x_star; θ...) + loglik(x_star, obs_lik) - logpdf(posterior, x_star)
end

# Implicit-Function-Theorem hyperparameter-gradient path for non-Gaussian latent priors.
# The real implementation (currently for `AutoDiffLatentPrior`) lives in the ForwardDiff
# extension; this stub fires only when θ already carries Dual partials but the extension
# isn't loaded — which can't normally happen — so it's a guard, not a user-facing error.
function _nongaussian_dualhp_ift(prior, obs_lik, θ_full, ws; kwargs...)
    throw(
        ArgumentError(
            "Hyperparameter (θ) gradients for $(typeof(prior)) require the ForwardDiff " *
                "extension to be loaded (`using ForwardDiff`)."
        )
    )
end

"""
    gaussian_approximation(prior::LatentModel, obs_lik; θ, ws=nothing, kwargs...)

Materialise the Gaussian latent prior at `θ` and delegate to the
`(::AbstractGMRF, obs_lik)` / `(::WorkspaceGMRF, obs_lik)` dispatch
for fixed-Q Newton.
"""
function gaussian_approximation(
        prior::LatentModel,
        obs_lik::ObservationLikelihood;
        θ::NamedTuple = NamedTuple(),
        ws::Union{Nothing, GMRFWorkspace} = nothing,
        x0 = nothing,
        max_iter::Int = 50,
        mean_change_tol::Real = 1.0e-4,
        newton_dec_tol::Real = 1.0e-5,
        adaptive_stepsize::Bool = true,
        max_linesearch_iter::Int = 10,
        verbose::Bool = false,
        hp_kwargs...,
    )
    θ_full = isempty(hp_kwargs) ? θ : merge(θ, NamedTuple(hp_kwargs))
    materialised = ws === nothing ? prior(; θ_full...) : prior(ws; θ_full...)
    return gaussian_approximation(
        materialised, obs_lik;
        x0, max_iter, mean_change_tol, newton_dec_tol,
        adaptive_stepsize, max_linesearch_iter, verbose,
    )
end

"""
    gaussian_approximation(prior::NonGaussianLatentPrior, obs_lik; θ, ws=nothing, x0, kwargs...)

Iterated-linearisation Gaussian approximation. The Newton loop calls
`local_quadratic(prior, x_k; θ...)` per iterate to re-linearise the
prior at the current iterate; the line-search merit uses the exact
`log p(x | θ)` carried in `LocalLatentQuadratic.logp_ref`.

Hyperparameter values may be passed splatted (`τ = 1.0`) or as a
`θ::NamedTuple` keyword; both forms merge before dispatch. Non-Gaussian
priors don't have a canonical mean, so `x0` defaults to
`zeros(length(prior))` — pass it explicitly for problems where zero is a
poor starting point (e.g. priors with reflection symmetry through zero).

When a `ws::GMRFWorkspace` is supplied, it must be seeded with the **full
structural sparsity pattern** that `local_quadratic(prior, x; θ...)`
produces across all Newton iterates (the workspace reuses one symbolic
factorisation, and the per-iterate precision values are copied onto it
positionally). Seeding from a generic, non-degenerate `x` is the safe way
to capture every structural coupling.
"""
function gaussian_approximation(
        prior::NonGaussianLatentPrior,
        obs_lik::ObservationLikelihood;
        θ::NamedTuple = NamedTuple(),
        ws::Union{Nothing, GMRFWorkspace} = nothing,
        x0 = nothing,
        max_iter::Int = 50,
        mean_change_tol::Real = 1.0e-4,
        newton_dec_tol::Real = 1.0e-5,
        adaptive_stepsize::Bool = true,
        max_linesearch_iter::Int = 10,
        verbose::Bool = false,
        hp_kwargs...,
    )
    θ_full = isempty(hp_kwargs) ? θ : merge(θ, NamedTuple(hp_kwargs))

    # Hyperparameter-gradient pass: when θ carries ForwardDiff.Dual partials, running the
    # Newton loop with Duals is impossible (CHOLMOD can't factorize a Dual matrix, and the
    # sparsity tracer can't coexist with Dual θ). Route to the Implicit Function Theorem
    # path (primal Newton + analytic θ-tangent), provided by the ForwardDiff extension.
    if _hp_carries_ad_partials(θ_full)
        return _nongaussian_dualhp_ift(
            prior, obs_lik, θ_full, ws;
            x0, max_iter, mean_change_tol, newton_dec_tol,
            adaptive_stepsize, max_linesearch_iter, verbose,
        )
    end

    lp = LatentPrior(prior, θ_full)
    constraint_info = constraints(prior; θ_full...)
    constraints_nt = constraint_info === nothing ? nothing :
        (A = constraint_info[1], e = constraint_info[2])
    x_init = x0 === nothing ? zeros(length(prior)) : copy(x0)

    if ws === nothing
        Q_seed, = _prior_local(lp, x_init)
        cache = deepcopy(linsolve_cache(GMRF(zeros(length(x_init)), Q_seed)))
        return _newton_loop(
            lp, obs_lik, cache, constraints_nt, x_init;
            max_iter, mean_change_tol, newton_dec_tol,
            adaptive_stepsize, max_linesearch_iter, verbose,
        )
    end

    return _workspace_newton_loop(
        lp, ws, obs_lik, constraints_nt, x_init;
        max_iter, mean_change_tol, newton_dec_tol,
        adaptive_stepsize, max_linesearch_iter, verbose,
    )
end
