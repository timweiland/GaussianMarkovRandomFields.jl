export ObservationIndependence, ConditionallyIndependent, ConditionallyDependent, observation_independence

"""
    ObservationIndependence

Abstract trait type for characterizing the conditional independence structure of observations
given the latent field.

See also: [`ConditionallyIndependent`](@ref), [`ConditionallyDependent`](@ref)
"""
abstract type ObservationIndependence end

"""
    ConditionallyIndependent <: ObservationIndependence

Trait indicating that observations are conditionally independent given the latent field `x`.

This means the joint log-likelihood can be decomposed as:
```math
\\log p(y | x) = \\sum_i \\log p(y_i | x_i)
```

Observation models with this trait support pointwise log-likelihood computation via
[`pointwise_loglik`](@ref).

# Examples
```julia
obs_model = ExponentialFamily(Poisson)
obs_lik = obs_model(y)
observation_independence(obs_lik)  # Returns ConditionallyIndependent()
```
"""
struct ConditionallyIndependent <: ObservationIndependence end

"""
    ConditionallyDependent <: ObservationIndependence

Trait indicating that observations have dependent structure given the latent field `x`.

This means the joint log-likelihood cannot be decomposed into a sum of independent
observation-wise terms. Examples include multivariate normal observations with correlated
noise.

Observation models with this trait do not support [`pointwise_loglik`](@ref).

# Note
No observation models in the current package have this trait. It is provided for future
extensibility and to make the conditional independence assumption explicit in the type
system.
"""
struct ConditionallyDependent <: ObservationIndependence end

"""
    observation_independence(obs_lik::ObservationLikelihood) -> ObservationIndependence

Return the conditional independence trait for an observation likelihood.

All current observation models return [`ConditionallyIndependent`](@ref). Future models
with correlated observations would return [`ConditionallyDependent`](@ref).

# Examples
```julia
obs_model = ExponentialFamily(Normal)
obs_lik = obs_model(y; σ=1.0)
observation_independence(obs_lik)  # Returns ConditionallyIndependent()
```

# Note
The default implementation is defined in `observation_likelihood.jl` after `ObservationLikelihood`
is defined, to avoid circular dependencies.
"""
function observation_independence end
