import DifferentiationInterface as DI

export AutoDiffLatentPrior

"""
    AutoDiffLatentPrior{F, B, SB, HN, C} <: NonGaussianLatentPrior

A non-Gaussian latent prior defined *implicitly* by a scalar joint log-density
`f(x; θ...) = log p(x | θ)`. The natural-form local quadratic
([`local_quadratic`](@ref)) is obtained by automatic differentiation: the
gradient `∇f` and a (sparse) Hessian `∇²f` give

    Q = -∇²f(x_ref),   h = ∇f(x_ref) + Q · x_ref,   logp_ref = f(x_ref).

This is the latent-side analogue of [`AutoDiffObservationModel`](@ref): it reuses
the same DifferentiationInterface backends, the eltype-keyed preparation cache,
and — with the `SparseADLikelihoods` extension loaded
(`SparseConnectivityTracer` + `SparseMatrixColorings`) — sparse-Hessian detection
and colouring. The Hessian sparsity is detected *structurally* (value-agnostic),
so the pattern is stable across Newton iterates, which is what lets the
`GMRFWorkspace` path reuse a single symbolic factorisation.

A monolithic TMB-style joint `f(x, θ; data)` is just an `AutoDiffLatentPrior`
carrying the entire energy, combined with a trivial likelihood — `marginal_loglikelihood`
then reduces to the Laplace marginal `log ∫ exp f(x, θ) dx`. The structured case
(an AD-defined latent prior plus an *exact* closed-form likelihood) works equally
well by composing this prior with any [`ObservationLikelihood`](@ref).

# Fields
- `logp_func::F`: log-density with signature `(x; θ...) -> Real`.
- `n::Int`: number of latent variables.
- `grad_backend::B` / `hess_backend::SB`: DI backends for `∇f` and `∇²f`.
- `prep_cache::_ADPrepCache`: eltype-keyed DI preparation cache.
- `hyperparams::HN`: tuple of hyperparameter names (`Symbol`s).
- `name::Symbol`: `model_name`, used for `CombinedModel` parameter prefixing.
- `constraints::C`: `nothing`, or a fixed `(A, e)` linear-equality constraint `A x = e`.
"""
struct AutoDiffLatentPrior{F, B, SB, HN, C} <: NonGaussianLatentPrior
    logp_func::F
    n::Int
    grad_backend::B
    hess_backend::SB
    prep_cache::_ADPrepCache
    hyperparams::HN
    name::Symbol
    constraints::C
end

"""
    AutoDiffLatentPrior(logp_func; n, hyperparams=(), grad_backend, hessian_backend, name=:autodiff, constraints=nothing)

Construct an `AutoDiffLatentPrior` wrapping the joint log-density
`logp_func(x; θ...) -> Real`. `hyperparams` is a tuple of the hyperparameter
names (`Symbol`s) the density expects as keyword arguments.

With the `SparseADLikelihoods` extension loaded, `hessian_backend` defaults to a
sparse backend (Tracer sparsity detection + greedy colouring); otherwise it falls
back to a dense Hessian (with a warning), which is fine for small problems.
"""
function AutoDiffLatentPrior(
        logp_func;
        n::Int,
        hyperparams = (),
        grad_backend = default_grad_backend(),
        hessian_backend = default_hessian_backend(grad_backend),
        name::Symbol = :autodiff,
        constraints = nothing,
    )
    if hessian_backend === nothing
        hessian_backend = grad_backend
        @warn "Hessian backend has type $(typeof(hessian_backend)) which may produce dense Hessians!! " *
            "Load the SparseADLikelihoods extension (SparseConnectivityTracer + SparseMatrixColorings) for sparse Hessians."
    end
    return AutoDiffLatentPrior(
        logp_func, n, grad_backend, hessian_backend, _ADPrepCache(n),
        Tuple(hyperparams), name, constraints,
    )
end

# --- AbstractLatentPrior interface ---
Base.length(m::AutoDiffLatentPrior) = m.n
model_name(m::AutoDiffLatentPrior) = m.name
constraints(m::AutoDiffLatentPrior; kwargs...) = m.constraints
hyperparameters(m::AutoDiffLatentPrior) =
    NamedTuple{m.hyperparams}(ntuple(_ -> Real, length(m.hyperparams)))

# Closure capturing the hyperparameters as a `NamedTuple`. For a given set of
# hyperparameter *types* this closure has a stable type, so the eltype-keyed prep
# cache reuses preparations across Newton iterates (the values may change).
_adlp_call(logp_func, θ::NamedTuple) = x -> logp_func(x; θ...)

# Compute eltype for the AD operator: widen `eltype(x)` by any AD-tagged
# hyperparameter value eltypes (so an outer Dual-θ pass prepares at the right
# type); plain primal θ leaves it at `eltype(x)`. Mirrors AutoDiffLikelihood.
function _adlp_compute_eltype(x, θ::NamedTuple)
    S = promote_type(eltype(x), map(_value_eltype, values(θ))...)
    return isconcretetype(S) ? S : eltype(x)
end

"""
    local_quadratic(m::AutoDiffLatentPrior, x_ref; θ...) -> LocalLatentQuadratic

Natural-form quadratic of the AD-defined log-density at `x_ref`, via one gradient
and one (sparse) Hessian evaluation: `Q = -∇²f`, `h = ∇f + Q·x_ref`,
`logp_ref = f(x_ref)`.
"""
function local_quadratic(m::AutoDiffLatentPrior, x_ref::AbstractVector; θ...)
    θnt = NamedTuple(θ)
    f = _adlp_call(m.logp_func, θnt)
    S = _adlp_compute_eltype(x_ref, θnt)
    xS = _as_eltype(S, x_ref)
    grad_prep = _get_or_prepare_grad!(m.prep_cache, f, m.grad_backend, S)
    hess_prep = _get_or_prepare_hess!(m.prep_cache, f, m.hess_backend, S)
    logp_ref = f(xS)
    g = DI.gradient(f, grad_prep, m.grad_backend, xS)
    H = DI.hessian(f, hess_prep, m.hess_backend, xS)
    Q = sparse(-H)
    h = g + Q * xS
    return LocalLatentQuadratic(Q, h, logp_ref, xS)
end

"""
    prior_logdensity(m::AutoDiffLatentPrior, x; θ...) -> Real

Direct primal evaluation `f(x; θ...)` — no AD, no Hessian. This is the cheap
line-search hook: backtracking calls it per trial point instead of rebuilding the
full local quadratic.
"""
prior_logdensity(m::AutoDiffLatentPrior, x::AbstractVector; θ...) = m.logp_func(x; θ...)
