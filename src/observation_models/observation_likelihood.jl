using SparseArrays
using Random
import DifferentiationInterface as DI

export ObservationLikelihood, loglik, loggrad, loghessian, pointwise_loglik, pointwise_loglik!

"""
    ObservationLikelihood

Abstract base type for materialized observation likelihoods.

Observation likelihoods are created by materializing an observation model with specific
hyperparameters θ and observed data y. They provide efficient evaluation methods that 
only depend on the latent field x, eliminating the need to repeatedly pass θ and y.

This design provides major performance benefits in optimization loops and cleaner 
automatic differentiation boundaries.

# Usage Pattern
```julia
# Step 1: Configure observation model (factory)
obs_model = ExponentialFamily(Normal)

# Step 2: Materialize with data and hyperparameters  
obs_lik = obs_model(y; σ=1.2)

# Step 3: Fast evaluation in hot loops
ll = loglik(x, obs_lik)      # Only x argument needed!
grad = loggrad(x, obs_lik)   # Fast x-only evaluation
```

"""
abstract type ObservationLikelihood end

# Default trait implementation: all likelihoods are conditionally independent unless overridden
observation_independence(::ObservationLikelihood) = ConditionallyIndependent()

loglik(x, obs_lik::ObservationLikelihood) = error("loglik is not implemented for $(typeof(obs_lik)).")
(obs_lik::ObservationLikelihood)(x) = loglik(x, obs_lik)

autodiff_gradient_backend(::ObservationLikelihood) = nothing
default_hessian_backend(grad_backend) = nothing
autodiff_hessian_backend(obs_lik::ObservationLikelihood) = default_hessian_backend(autodiff_gradient_backend(obs_lik))
autodiff_gradient_prep(::ObservationLikelihood) = nothing
autodiff_hessian_prep(::ObservationLikelihood) = nothing


"""
    loggrad(x, obs_lik::ObservationLikelihood) -> Vector{Float64}

Automatic differentiation fallback for ObservationLikelihood gradient computation.
"""
function loggrad(x, obs_lik::ObservationLikelihood)
    backend = autodiff_gradient_backend(obs_lik)
    return if backend !== nothing
        grad_prep = autodiff_gradient_prep(obs_lik)
        if grad_prep !== nothing
            return DI.gradient(obs_lik, grad_prep, backend, x)
        else
            return DI.gradient(obs_lik, backend, x)
        end
    else
        obs_lik_type = typeof(obs_lik)
        error(
            "loggrad not implemented for $(obs_lik_type).\n"
                * "Try implementing `autodiff_gradient_backend(::$(obs_lik_type))`."
        )
    end
end

"""
    loghessian(x, obs_lik::ObservationLikelihood) -> AbstractMatrix{Float64}

Automatic differentiation fallback for ObservationLikelihood Hessian computation.
"""
function loghessian(x, obs_lik::ObservationLikelihood)
    backend = autodiff_hessian_backend(obs_lik)
    if backend === nothing
        backend = autodiff_gradient_backend(obs_lik)
    end
    return if backend !== nothing
        hessian_prep = autodiff_hessian_prep(obs_lik)
        if hessian_prep !== nothing
            return DI.hessian(obs_lik, hessian_prep, backend, x)
        else
            return DI.hessian(obs_lik, backend, x)
        end
    else
        obs_lik_type = typeof(obs_lik)
        error(
            "loghessian not implemented for $(obs_lik_type).\n"
                * "Try implementing `autodiff_hessian_backend(::$(obs_lik_type))`."
        )
    end
end

"""
    pointwise_loglik(x, obs_lik::ObservationLikelihood) -> Vector{Float64}

Compute log-likelihood for each observation individually.

For conditionally independent observations, returns a vector where `result[i] = log p(yᵢ | xᵢ)`.
The sum of all elements equals the total log-likelihood (up to numerical precision):

```julia
sum(pointwise_loglik(x, obs_lik)) ≈ loglik(x, obs_lik)
```

# Arguments
- `x::AbstractVector`: Latent field values
- `obs_lik::ObservationLikelihood`: Materialized observation likelihood

# Returns
- `Vector{Float64}`: Per-observation log-likelihoods of length `n_obs`

# Conditional Independence Requirement

This function requires observations to be conditionally independent given the latent field.
Check this property via `observation_independence(obs_lik)`:

```julia
obs_independence = observation_independence(obs_lik)
obs_independence isa ConditionallyIndependent  # Required for pointwise_loglik
```

Observation models with [`ConditionallyDependent`](@ref) trait will error.

# Use Cases

Primary use case is computing model comparison metrics that require per-observation
log-likelihoods:
- **WAIC** (Widely Applicable Information Criterion)
- **LOO-CV** (Leave-One-Out Cross-Validation)
- **CPO** (Conditional Predictive Ordinate)

These metrics require storing pointwise log-likelihoods to enable post-hoc computation
of information criteria.

# Examples
```julia
# Basic usage
obs_model = ExponentialFamily(Poisson)
obs_lik = obs_model(y)
x = randn(length(y))

# Per-observation log-likelihoods
per_obs = pointwise_loglik(x, obs_lik)  # Vector of length n
total = loglik(x, obs_lik)              # Scalar

@assert sum(per_obs) ≈ total
@assert length(per_obs) == length(y)

# With observation indices (subset of latent field)
obs_model = ExponentialFamily(Normal; indices=[1, 5, 10])
obs_lik = obs_model(y_subset; σ=1.0)
x_full = randn(20)

per_obs = pointwise_loglik(x_full, obs_lik)  # Length 3, not 20!
```

# Implementation Note

For custom observation likelihoods, implement the internal method:
```julia
_pointwise_loglik(::ConditionallyIndependent, x, obs_lik::MyLikelihood)
```

See also: [`pointwise_loglik!`](@ref), [`observation_independence`](@ref)
"""
function pointwise_loglik(x, obs_lik::ObservationLikelihood)
    return _pointwise_loglik(observation_independence(obs_lik), x, obs_lik)
end

"""
    pointwise_loglik!(result, x, obs_lik::ObservationLikelihood) -> Vector{Float64}

In-place version of [`pointwise_loglik`](@ref).

Computes per-observation log-likelihoods and stores them in `result`, which must have
length equal to the number of observations.

# Arguments
- `result::AbstractVector{Float64}`: Pre-allocated output vector of length `n_obs`
- `x::AbstractVector`: Latent field values
- `obs_lik::ObservationLikelihood`: Materialized observation likelihood

# Returns
- `result`: The same vector passed in, now filled with per-observation log-likelihoods

# Examples
```julia
obs_model = ExponentialFamily(Poisson)
obs_lik = obs_model(y)
x = randn(length(y))

# Pre-allocate output
result = zeros(length(y))

# In-place computation
pointwise_loglik!(result, x, obs_lik)

@assert sum(result) ≈ loglik(x, obs_lik)
```

# Performance

Use this version in hot loops where repeated allocation would be problematic:
```julia
# Example: computing metrics across multiple latent field samples
result = zeros(n_obs)
for i in 1:n_samples
    x = sample_latent_field()
    pointwise_loglik!(result, x, obs_lik)
    # ... use result ...
end
```

See also: [`pointwise_loglik`](@ref)
"""
function pointwise_loglik!(result, x, obs_lik::ObservationLikelihood)
    return _pointwise_loglik!(observation_independence(obs_lik), result, x, obs_lik)
end

# Internal trait-dispatched implementations

"""
    _pointwise_loglik(independence_trait, x, obs_lik)

Internal method for trait-based dispatch of pointwise log-likelihood computation.

Custom observation likelihoods should implement:
```julia
function _pointwise_loglik(::ConditionallyIndependent, x, obs_lik::MyLikelihood)
    # Return vector of per-observation log-likelihoods
end
```
"""
function _pointwise_loglik(::ConditionallyDependent, x, obs_lik::ObservationLikelihood)
    obs_lik_type = typeof(obs_lik)
    error(
        "pointwise_loglik not supported for observation model with correlated observations.\n"
            * "$(obs_lik_type) has trait ConditionallyDependent().\n"
            * "Pointwise log-likelihoods are only well-defined for conditionally independent observations."
    )
end

function _pointwise_loglik(::ConditionallyIndependent, x, obs_lik::ObservationLikelihood)
    obs_lik_type = typeof(obs_lik)
    error(
        "pointwise_loglik not implemented for $(obs_lik_type).\n"
            * "Implement `_pointwise_loglik(::ConditionallyIndependent, x, ::$(obs_lik_type))`."
    )
end

"""
    _pointwise_loglik!(independence_trait, result, x, obs_lik)

Internal method for trait-based dispatch of in-place pointwise log-likelihood computation.

Custom observation likelihoods should implement:
```julia
function _pointwise_loglik!(::ConditionallyIndependent, result, x, obs_lik::MyLikelihood)
    # Fill result with per-observation log-likelihoods
    return result
end
```
"""
function _pointwise_loglik!(::ConditionallyDependent, result, x, obs_lik::ObservationLikelihood)
    obs_lik_type = typeof(obs_lik)
    error(
        "pointwise_loglik! not supported for observation model with correlated observations.\n"
            * "$(obs_lik_type) has trait ConditionallyDependent().\n"
            * "Pointwise log-likelihoods are only well-defined for conditionally independent observations."
    )
end

function _pointwise_loglik!(::ConditionallyIndependent, result, x, obs_lik::ObservationLikelihood)
    obs_lik_type = typeof(obs_lik)
    error(
        "pointwise_loglik! not implemented for $(obs_lik_type).\n"
            * "Implement `_pointwise_loglik!(::ConditionallyIndependent, result, x, ::$(obs_lik_type))`."
    )
end
