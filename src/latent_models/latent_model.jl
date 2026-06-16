export AbstractLatentPrior, LatentModel, NonGaussianLatentPrior,
    hyperparameters, precision_matrix, mean, constraints, model_name

"""
    AbstractLatentPrior

Top of the latent-prior hierarchy. Anything that can be the prior side of
`gaussian_approximation` is an `AbstractLatentPrior`. Two children:

- [`LatentModel`](@ref): a Gaussian latent prior. `(Q, μ)` are determined
  by the hyperparameters alone, so the prior materialises as a `GMRF`.
- [`NonGaussianLatentPrior`](@ref): a latent prior whose log-density is
  not quadratic in `x`. There is no canonical materialised `GMRF`; the
  prior is only meaningful at a chosen reference point through
  [`local_quadratic`](@ref).

Every concrete subtype must implement:
- `length(prior)` — number of latent variables.
- `hyperparameters(prior)` — names/types of hyperparameters.
- `model_name(prior)` — `Symbol` used for `CombinedModel` parameter prefixing.
- `constraints(prior; θ...)` — `nothing` or `(A, e)`.
"""
abstract type AbstractLatentPrior end

"""
    LatentModel <: AbstractLatentPrior

Abstract type for *Gaussian* latent priors. `(Q, μ)` depend only on the
hyperparameters `θ`, so the prior materialises as a `GMRF` /
`ConstrainedGMRF`. In addition to the [`AbstractLatentPrior`](@ref)
interface, every concrete subtype must implement:

- `precision_matrix(model; θ...)`
- `mean(model; θ...)`
- `(model)(; θ...) -> AbstractGMRF`

`local_quadratic(::LatentModel, x_ref; θ...)` has a default that uses
these methods together with `logpdf`, so subtypes don't need to override
it for the iterated path.

# Usage
```julia
model = AR1Model(100)
gmrf = model(; τ = 2.0, ρ = 0.8)   # materialise a GMRF
```
"""
abstract type LatentModel <: AbstractLatentPrior end

"""
    NonGaussianLatentPrior <: AbstractLatentPrior

Abstract type for latent priors whose `log p(x | θ)` is *not* quadratic
in `x`. There is no canonical materialised `GMRF`; the prior is queried
through [`local_quadratic`](@ref) at a reference point `x_ref`. In
addition to the [`AbstractLatentPrior`](@ref) interface, every concrete
subtype must implement:

- `local_quadratic(prior, x_ref; θ...) -> LocalLatentQuadratic`

Models of this type cannot be called as `prior(; θ...)` because
materialisation is x-dependent. Use `gaussian_approximation(prior, obs_lik; θ...)`
to drive the iterated-linearisation Newton; pass `x0` explicitly if
zeros aren't an appropriate starting point.
"""
abstract type NonGaussianLatentPrior <: AbstractLatentPrior end

"""
    length(prior::AbstractLatentPrior) -> Int

Number of latent variables in the prior.
"""
Base.length(prior::AbstractLatentPrior) = throw(MethodError(length, (prior,)))

"""
    hyperparameters(prior::AbstractLatentPrior) -> NamedTuple

`NamedTuple` describing the hyperparameter names and their expected types.
"""
hyperparameters(prior::AbstractLatentPrior) = throw(MethodError(hyperparameters, (prior,)))

"""
    constraints(prior::AbstractLatentPrior; θ...) -> Union{Nothing, Tuple}

Linear-equality constraint information for the prior at hyperparameters
`θ`. Either `nothing` (unconstrained) or a tuple `(A, e)` such that
`A x = e` is enforced.
"""
constraints(prior::AbstractLatentPrior; kwargs...) = throw(MethodError(constraints, (prior,)))

"""
    model_name(prior::AbstractLatentPrior) -> Symbol

Symbol used as a parameter-name suffix when this prior is composed with
others in a `CombinedModel` (so e.g. `τ` from two priors becomes
`τ_ar1` and `τ_besag`).
"""
model_name(prior::AbstractLatentPrior) = throw(MethodError(model_name, (prior,)))

"""
    precision_matrix(model::LatentModel; θ...) -> AbstractMatrix

Precision matrix of the Gaussian latent prior at hyperparameters `θ`.
"""
precision_matrix(model::LatentModel; kwargs...) = throw(MethodError(precision_matrix, (model,)))

"""
    mean(model::LatentModel; θ...) -> AbstractVector

Mean vector of the Gaussian latent prior at hyperparameters `θ`.
"""
mean(model::LatentModel; kwargs...) = throw(MethodError(mean, (model,)))

"""
    (model::LatentModel)(; θ...) -> AbstractGMRF

Materialise a Gaussian `LatentModel` at hyperparameters `θ`. Returns a
`GMRF` if `constraints(model; θ...) === nothing`, otherwise a
`ConstrainedGMRF`.
"""
function (model::LatentModel)(; kwargs...)
    μ = mean(model; kwargs...)
    Q = precision_matrix(model; kwargs...)
    constraint_info = constraints(model; kwargs...)

    if constraint_info === nothing
        return GMRF(μ, Q, model.alg)
    else
        A, e = constraint_info
        base_gmrf = GMRF(μ, Q, model.alg)
        return ConstrainedGMRF(base_gmrf, A, e)
    end
end
