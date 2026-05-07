import DifferentiationInterface as DI
using LinearAlgebra: Diagonal

export AutoDiffLikelihood, AutoDiffObservationModel

"""
    _ADPrepCache

Mutable, eltype-keyed cache of `DI.prepare_gradient` / `DI.prepare_hessian`
results. A single `AutoDiffLikelihood` holds one of these so repeated calls
with the same input eltype reuse the prep (the common case), while
unexpected eltypes (e.g. AD-tagged scalars from a nested-AD caller) trigger
a one-time prep on first miss instead of failing.

`get!` on the underlying `Dict`s isn't safe under concurrent mutation, so
the cache holds a `ReentrantLock`. The prep itself can be expensive; we
don't try to avoid duplicated work under contention beyond the lock.

For AD backends with type-level perturbation tracking (e.g. ForwardDiff's
`Dual{Tag, V, N}`), the cache key carries the full tagged eltype. This is
structurally necessary — the prep's internal buffers bake the tagged
scalar type into their own value-type, so two distinct outer-AD passes
produce incompatible prep buffers. Stripping the tag from the key would
point at a prep the backend's safety checks would reject. Practical
consequence: each outer AD call site adds one cache entry. With a hoisted
outer function the tag is stable across calls and the cache stays at 1–2
entries; with a fresh closure per iteration the cache grows by one entry
per iteration. Hoist outer closures to top-level functions to avoid this.
"""
mutable struct _ADPrepCache
    n_latent::Int
    grad_preps::Dict{DataType, Any}
    hess_preps::Dict{DataType, Any}
    lock::ReentrantLock
end

_ADPrepCache(n_latent::Int) = _ADPrepCache(n_latent, Dict{DataType, Any}(), Dict{DataType, Any}(), ReentrantLock())

function _get_or_prepare_grad!(cache::_ADPrepCache, loglik_func, backend, ::Type{T}) where {T}
    return @lock cache.lock get!(cache.grad_preps, T) do
        DI.prepare_gradient(loglik_func, backend, zeros(T, cache.n_latent))
    end
end

function _get_or_prepare_hess!(cache::_ADPrepCache, loglik_func, backend, ::Type{T}) where {T}
    return @lock cache.lock get!(cache.hess_preps, T) do
        DI.prepare_hessian(loglik_func, backend, zeros(T, cache.n_latent))
    end
end

"""
    AutoDiffLikelihood{F, B, SB, PF, Y, HP} <: ObservationLikelihood

Automatic differentiation-based observation likelihood that wraps a user-provided
log-likelihood function.

The likelihood stores `y` (data) and `hyperparams` (a `NamedTuple` of
hyperparameter values) as separate fields rather than baking them into a
closure. At evaluation time the stored values are splatted as keyword
arguments to `loglik_func`, so the user's function signature
`(x; y, hyperparam_kwargs...) -> Real` is unchanged.

Storing hyperparameters explicitly (rather than capturing them in a
closure) is what makes nested-AD scenarios work cleanly: when a
hyperparameter carries AD partials from an outer pass, the AD-tagged-ness
is visible in `HP`'s type parameter, and AD-backend extensions can detect
it via dispatch and route `gaussian_approximation` through a primal-Newton
+ manual-θ-tangent flow without nesting AD inside the Newton iteration.

# Type Parameters
- `F`: Type of the log-likelihood function
- `B`: Type of the AD backend for gradients
- `SB`: Type of the AD backend for Hessians
- `PF`: Type of the pointwise log-likelihood function (`Union{Nothing, Function}`)
- `Y`: Type of the observation data
- `HP`: `NamedTuple` type of stored hyperparameters

# Fields
- `loglik_func::F`: User log-likelihood with signature `(x; y, hyperparam_kwargs...) -> Real`.
- `grad_backend::B`: AD backend for gradient computation.
- `hess_backend::SB`: AD backend for Hessian computation.
- `prep_cache::_ADPrepCache`: Eltype-keyed DI prep cache. Float64 entry populated eagerly.
- `pointwise_loglik_func::PF`: Optional pointwise log-likelihood, signature `(x; y, hyperparam_kwargs...) -> Vector{Real}`.
- `y::Y`: Stored observation data.
- `hyperparams::HP`: Stored hyperparameter values.
"""
struct AutoDiffLikelihood{F, B, SB, PF, Y, HP <: NamedTuple} <: GaussianMarkovRandomFields.ObservationLikelihood
    loglik_func::F
    grad_backend::B
    hess_backend::SB
    prep_cache::_ADPrepCache
    pointwise_loglik_func::PF
    y::Y
    hyperparams::HP
end

"""
    AutoDiffObservationModel{F, B, SB, H, PF} <: ObservationModel

Observation model factory for `AutoDiffLikelihood`. Stores the parent log-likelihood
function and AD backend choices; calling the model with data and hyperparameters
materializes a likelihood that holds those values explicitly.

# Type Parameters
- `F`: Type of the log-likelihood function
- `B`: Type of the AD backend for gradients
- `SB`: Type of the AD backend for Hessians
- `H`: Type of the hyperparameter-name tuple
- `PF`: Type of the pointwise log-likelihood function (`Union{Nothing, Function}`)

# Fields
- `loglik_func::F`: User log-likelihood with signature `(x; y, hyperparam_kwargs...) -> Real`.
- `n_latent::Int`: Number of latent field components.
- `grad_backend::B`: AD backend for gradients.
- `hess_backend::SB`: AD backend for Hessians.
- `hyperparams::H`: Tuple of hyperparameter names.
- `pointwise_loglik_func::PF`: Optional pointwise log-likelihood.

# Usage
```julia
function my_loglik(x; σ, y)
    return -0.5 * sum((y .- x).^2) / σ^2 - length(y) * log(σ)
end
obs_model = AutoDiffObservationModel(my_loglik; n_latent=2, hyperparams=(:σ,))
obs_lik  = obs_model([1.2, 1.8]; σ=0.5)   # materialize with data + hyperparams
loglik(x, obs_lik)
loggrad(x, obs_lik)
loghessian(x, obs_lik)
```
"""
struct AutoDiffObservationModel{F, B, SB, H, PF} <: GaussianMarkovRandomFields.ObservationModel
    loglik_func::F
    n_latent::Int
    grad_backend::B
    hess_backend::SB
    hyperparams::H
    pointwise_loglik_func::PF
end

const AD_PREFERRED_ORDER = (DI.AutoEnzyme(), DI.AutoMooncake(), DI.AutoZygote(), DI.AutoForwardDiff())
function default_grad_backend()
    ad_idx = findfirst(DI.check_available, AD_PREFERRED_ORDER)
    if ad_idx === nothing
        error(
            "None of the default AD backends are available."
                * " Please specify your own grad_backend."
        )
    end
    return AD_PREFERRED_ORDER[ad_idx]
end

"""
    AutoDiffObservationModel(loglik_func; n_latent, hyperparams=(), grad_backend, hessian_backend, pointwise_loglik_func=nothing)

Construct an `AutoDiffObservationModel`.

`loglik_func` should have signature `(x; y, hyperparam_kwargs...) -> Real`.
`hyperparams` is a tuple of hyperparameter names (`Symbol`s) the model expects.
"""
function AutoDiffObservationModel(loglik_func; n_latent, hyperparams = (), grad_backend = default_grad_backend(), hessian_backend = default_hessian_backend(grad_backend), pointwise_loglik_func = nothing)
    if hessian_backend === nothing
        hessian_backend = grad_backend
        @warn "Hessian backend has type $(typeof(hessian_backend)) which may produce dense Hessians!!"
    end
    return AutoDiffObservationModel(loglik_func, n_latent, grad_backend, hessian_backend, hyperparams, pointwise_loglik_func)
end

"""
    AutoDiffLikelihood(loglik_func; n_latent, y, hyperparams, grad_backend, hessian_backend, pointwise_loglik_func)
    AutoDiffLikelihood(closure;     n_latent, grad_backend, hessian_backend, pointwise_loglik_func)

Direct construction of an `AutoDiffLikelihood`.

Two forms:

1. **Explicit `y` + `hyperparams`** (preferred): pass the parent `loglik_func`
   together with `y` and a `hyperparams::NamedTuple`. The likelihood stores them
   as separate fields, which lets the AD layer detect Dual-valued
   hyperparameters at the type level.

2. **Bare closure** (legacy): pass a function of `x` only, with no
   `y`/`hyperparams` arguments. Stored with empty `y` and empty `hyperparams`.
   Convenient for one-off uses but loses the nested-AD-friendly dispatch
   route — prefer form (1) when differentiating through a hyperparameter.
"""
function AutoDiffLikelihood(
        loglik_func;
        n_latent,
        y = nothing,
        hyperparams::NamedTuple = NamedTuple(),
        grad_backend = default_grad_backend(),
        hessian_backend = default_hessian_backend(grad_backend),
        pointwise_loglik_func = nothing,
    )
    if hessian_backend === nothing
        hessian_backend = grad_backend
        @warn "Hessian backend has type $(typeof(hessian_backend)) which may produce dense Hessians!!"
    end
    obs_lik = AutoDiffLikelihood(
        loglik_func, grad_backend, hessian_backend, _ADPrepCache(n_latent),
        pointwise_loglik_func, y, hyperparams
    )
    # Eagerly populate the Float64 prep entry. Construction and runtime calls
    # both go through `_build_call(obs_lik)` so the prep is keyed on the same
    # callable type as future evaluations — DI's PreparationMismatchError
    # check on closure identity is satisfied.
    f = _build_call(obs_lik)
    _get_or_prepare_grad!(obs_lik.prep_cache, f, grad_backend, Float64)
    _get_or_prepare_hess!(obs_lik.prep_cache, f, hessian_backend, Float64)
    return obs_lik
end

"""
    _strip_ad_partials_hyperparams(hp::NamedTuple) -> NamedTuple

Strip any AD-partial-carrying values from a hyperparameter `NamedTuple`,
returning the primal-valued counterpart. No-op for non-AD entries. Used at
construction time to size the Float64 prep cache correctly even when the
caller already has AD-tagged hyperparams in scope.
"""
_strip_ad_partials_hyperparams(hp::NamedTuple) = NamedTuple{keys(hp)}(map(_strip_ad_partials, values(hp)))
_strip_ad_partials(x) = x

# Detection of AD-partial sensitivity in stored hyperparams. The actual
# recognition methods live in AD-backend extensions (currently the
# ForwardDiff ext, which extends `_carries_ad_partials` for `Dual`
# scalars and arrays). Main src just provides the default (no-AD)
# fallback so this works without any AD backend loaded.
"""
    _hp_carries_ad_partials(hp::NamedTuple) -> Bool

True if any element of `hp` carries AD partial information (e.g. a
`ForwardDiff.Dual`). AD-backend extensions override `_carries_ad_partials`
for the relevant scalar/array types; the default returns `false` so the
main src doesn't need to know about any specific AD backend.
"""
_hp_carries_ad_partials(hp::NamedTuple) = any(_carries_ad_partials, values(hp))
_carries_ad_partials(::Any) = false

function (obs_model::AutoDiffObservationModel)(y; kwargs...)
    hyperparams = NamedTuple(kwargs)
    return AutoDiffLikelihood(
        obs_model.loglik_func;
        n_latent = obs_model.n_latent,
        y = y,
        hyperparams = hyperparams,
        grad_backend = obs_model.grad_backend,
        hessian_backend = obs_model.hess_backend,
        pointwise_loglik_func = obs_model.pointwise_loglik_func,
    )
end

# =======================================================================================
# OBSERVATION MODEL INTERFACE IMPLEMENTATION
# =======================================================================================
latent_dimension(obs_model::AutoDiffObservationModel, y::AbstractVector) = obs_model.n_latent
hyperparameters(obs_model::AutoDiffObservationModel) = obs_model.hyperparams

# =======================================================================================
# CORE LIKELIHOOD EVALUATION METHODS
# =======================================================================================

"""
    _build_call(obs_lik) -> Function

Materialize a call closure `x -> loglik_func(x; y, hyperparams...)` from the
stored fields. Re-built each evaluation; the closure type changes with the
hyperparam eltype, which is what the prep cache keys on.

Bare-closure-form likelihoods (constructed without `y`/`hyperparams`)
short-circuit to the stored `loglik_func` directly.

Note: each `_build_call` call returns a fresh closure *object*, but two
closures produced from the same `obs_lik` share a generated closure
*type* (Julia gives one type per closure expression at the same source
location, parameterised by the captures' types). DI's PreparationMismatch
check is type-based, so reusing a prep prepared with an earlier
`_build_call(obs_lik)` against a later one is safe under current Julia
semantics. If Julia ever changes closure-type generation to depend on
captured-value identity rather than just captured-type, this assumption
needs revisiting.
"""
function _build_call(obs_lik::AutoDiffLikelihood)
    if obs_lik.y === nothing && isempty(obs_lik.hyperparams)
        return obs_lik.loglik_func
    end
    return x -> obs_lik.loglik_func(x; y = obs_lik.y, obs_lik.hyperparams...)
end

function _build_pointwise_call(obs_lik::AutoDiffLikelihood)
    obs_lik.pointwise_loglik_func === nothing && return nothing
    if obs_lik.y === nothing && isempty(obs_lik.hyperparams)
        return obs_lik.pointwise_loglik_func
    end
    return x -> obs_lik.pointwise_loglik_func(x; y = obs_lik.y, obs_lik.hyperparams...)
end

"""
    loglik(x, obs_lik::AutoDiffLikelihood) -> Real

Evaluate the stored log-likelihood at latent field `x`.
"""
function loglik(x, obs_lik::AutoDiffLikelihood)
    return _build_call(obs_lik)(x)
end

function loggrad(x, obs_lik::AutoDiffLikelihood)
    f = _build_call(obs_lik)
    prep = _get_or_prepare_grad!(obs_lik.prep_cache, f, obs_lik.grad_backend, eltype(x))
    return DI.gradient(f, prep, obs_lik.grad_backend, x)
end

function loghessian(x, obs_lik::AutoDiffLikelihood)
    if obs_lik.pointwise_loglik_func !== nothing
        return _diagonal_hessian_via_pointwise(x, obs_lik)
    end
    f = _build_call(obs_lik)
    prep = _get_or_prepare_hess!(obs_lik.prep_cache, f, obs_lik.hess_backend, eltype(x))
    return DI.hessian(f, prep, obs_lik.hess_backend, x)
end

# Pointwise structured-Hessian fast path. For conditionally-independent
# observation likelihoods (which `pointwise_loglik_func` declares):
#
#   loglik(x) = sum_i pointwise[i](x_i; y_i, θ)
#
# the Hessian is diagonal with `H[i,i] = ∂²pointwise[i]/∂x_i²`. Each entry
# is a 1D second derivative — the cleanest case for nested AD — so this
# path also doubles as the friction-free route for outer-AD callers (no
# DI nested-Dual prep machinery to navigate).
function _diagonal_hessian_via_pointwise(x, obs_lik::AutoDiffLikelihood)
    pf = _build_pointwise_call(obs_lik)
    n = length(x)
    backend = obs_lik.hess_backend
    diag_vals = map(1:n) do i
        # Closure that returns the i-th pointwise contribution as a function
        # of x[i] alone. `pf(x_perturbed)` computes the full vector, but only
        # the i-th entry depends on the perturbed x[i] — others are wasted
        # but correct. (Acceptable cost given this only runs when the dense
        # DI.hessian path would be worse, e.g. nested-AD scenarios.)
        # The comprehension promotes eltype to whatever `xi` is, so this
        # works under outer AD that injects Dual scalars into the closure.
        f_i = function (xi)
            x_perturbed = [j == i ? xi : x[j] for j in 1:n]
            return pf(x_perturbed)[i]
        end
        DI.second_derivative(f_i, backend, x[i])
    end
    return Diagonal(diag_vals)
end

# =======================================================================================
# AUTODIFF INTERFACE IMPLEMENTATION
# =======================================================================================
autodiff_gradient_backend(obs_lik::AutoDiffLikelihood) = obs_lik.grad_backend
autodiff_hessian_backend(obs_lik::AutoDiffLikelihood) = obs_lik.hess_backend
autodiff_gradient_prep(obs_lik::AutoDiffLikelihood) = obs_lik.prep_cache.grad_preps[Float64]
autodiff_hessian_prep(obs_lik::AutoDiffLikelihood) = obs_lik.prep_cache.hess_preps[Float64]

# =======================================================================================
# POINTWISE LOG-LIKELIHOOD IMPLEMENTATION
# =======================================================================================

function _pointwise_loglik(::ConditionallyIndependent, x, obs_lik::AutoDiffLikelihood)
    if obs_lik.pointwise_loglik_func === nothing
        error(
            "pointwise_loglik not available for this AutoDiffLikelihood.\n"
                * "To enable pointwise log-likelihood computation, provide the `pointwise_loglik_func` keyword argument\n"
                * "when constructing AutoDiffObservationModel:\n\n"
                * "    obs_model = AutoDiffObservationModel(loglik_func;\n"
                * "                                          n_latent=...,\n"
                * "                                          pointwise_loglik_func=my_pointwise_func)\n\n"
                * "The pointwise function should have signature `(x; y, hyperparam_kwargs...) -> Vector{Real}` where\n"
                * "result[i] = log p(yᵢ | xᵢ) and sum(result) ≈ loglik_func(x; y, hyperparam_kwargs...)."
        )
    end
    return _build_pointwise_call(obs_lik)(x)
end

function _pointwise_loglik!(::ConditionallyIndependent, result, x, obs_lik::AutoDiffLikelihood)
    per_obs = _pointwise_loglik(ConditionallyIndependent(), x, obs_lik)
    copyto!(result, per_obs)
    return result
end

# =======================================================================================
# PRETTY PRINTING
# =======================================================================================

# COV_EXCL_START
function Base.show(io::IO, obs_model::AutoDiffObservationModel)
    func_name = string(obs_model.loglik_func)
    if occursin("var\"", func_name) || occursin("#", func_name)
        func_name = "user function"
    end
    hyperparams_str = isempty(obs_model.hyperparams) ? "none" : join(obs_model.hyperparams, ", ")
    print(io, "AutoDiffObservationModel(")
    print(io, func_name)
    print(io, "; n_latent=", obs_model.n_latent)
    print(io, ", hyperparams=(", hyperparams_str, ")")
    print(io, ", grad_backend=", Base.typename(typeof(obs_model.grad_backend)).wrapper)
    print(io, ", hess_backend=", Base.typename(typeof(obs_model.hess_backend)).wrapper)
    return print(io, ")")
end

function Base.show(io::IO, obs_lik::AutoDiffLikelihood)
    func_name = string(obs_lik.loglik_func)
    if occursin("var\"", func_name) || occursin("#", func_name)
        func_name = "user function"
    end
    print(io, "AutoDiffLikelihood(")
    print(io, func_name)
    if !isempty(obs_lik.hyperparams)
        print(io, "; hyperparams=(", join(keys(obs_lik.hyperparams), ", "), ")")
    end
    print(io, ", grad_backend=", Base.typename(typeof(obs_lik.grad_backend)).wrapper)
    print(io, ", hess_backend=", Base.typename(typeof(obs_lik.hess_backend)).wrapper)
    return print(io, ")")
end
# COV_EXCL_STOP
