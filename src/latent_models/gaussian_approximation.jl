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
    lq_star = local_quadratic(prior, x_star; θ...)
    return lq_star.logp_ref + loglik(x_star, obs_lik) - logpdf(posterior, x_star)
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
    lp = LatentPrior(prior, θ_full)
    constraint_info = constraints(prior; θ_full...)
    constraints_nt = constraint_info === nothing ? nothing :
        (A = constraint_info[1], e = constraint_info[2])
    x_init = x0 === nothing ? zeros(length(prior)) : copy(x0)

    if ws === nothing
        Q_seed = prior_quadratic(lp, x_init).Q
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
