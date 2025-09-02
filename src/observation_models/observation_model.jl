using Random

export ObservationModel, hyperparameters, latent_dimension, conditional_distribution

"""
    ObservationModel

Abstract base type for all observation models for GMRFs.

An observation model defines the relationship between observations `y` and the latent field `x`,
typically through a likelihood function. ObservationModel types serve as factories for creating
ObservationLikelihood instances via callable syntax.

# Usage Pattern
```julia
# Step 1: Create observation model (factory)
obs_model = ExponentialFamily(Normal)

# Step 2: Materialize with data and hyperparameters
obs_lik = obs_model(y; σ=1.2)  # Creates ObservationLikelihood

# Step 3: Use materialized likelihood in hot loops
ll = loglik(x, obs_lik)  # Fast x-only evaluation
```

See also: [`ObservationLikelihood`](@ref), [`ExponentialFamily`](@ref)
"""
abstract type ObservationModel end


"""
    hyperparameters(obs_model::ObservationModel) -> Tuple{Vararg{Symbol}}

Return a tuple of required hyperparameter names for this observation model.

This method defines which hyperparameters the observation model expects to receive
when materializing an ObservationLikelihood instance.

# Arguments
- `obs_model`: An observation model implementing the `ObservationModel` interface

# Returns
- `Tuple{Vararg{Symbol}}`: Tuple of parameter names (e.g., `(:σ,)` or `(:α, :β)`)

# Example
```julia
hyperparameters(ExponentialFamily(Normal)) == (:σ,)
hyperparameters(ExponentialFamily(Bernoulli)) == ()
```

# Implementation
All observation models should implement this method. The default returns an empty tuple.
"""
hyperparameters(obs_model::ObservationModel) = ()

"""
    latent_dimension(obs_model::ObservationModel, y::AbstractVector) -> Union{Int, Nothing}

Return the latent field dimension for this observation model given observations y.

For most observation models, this will be `length(y)` (1:1 mapping).
For transformed observation models like `LinearlyTransformedObservationModel`,
this will be the dimension of the design matrix.

Returns `nothing` if the latent dimension cannot be determined automatically.

# Arguments
- `obs_model`: An observation model implementing the `ObservationModel` interface
- `y`: Vector of observations

# Returns
- `Int`: The latent field dimension, or `nothing` if unknown

# Example
```julia
latent_dimension(ExponentialFamily(Normal), y) == length(y)
latent_dimension(LinearlyTransformedObservationModel(base, A), y) == size(A, 2)
```
"""
latent_dimension(obs_model::ObservationModel, y::AbstractVector) = nothing

"""
    conditional_distribution(obs_model::ObservationModel, x; kwargs...) -> Distribution

Construct the conditional distribution p(y | x, θ) for sampling new observations.

This function returns a Distribution object that represents the probability 
distribution over observations y given latent field values x and hyperparameters θ.
It is used for sampling new observations from the observation model.

# Arguments
- `obs_model`: An observation model implementing the `ObservationModel` interface
- `x`: Latent field values (vector)  
- `kwargs...`: Hyperparameters as keyword arguments

# Returns
Distribution object that can be used with `rand()` to generate observations

# Example
```julia
model = ExponentialFamily(Poisson)
x = [1.0, 2.0]
dist = conditional_distribution(model, x)  # Poisson has no hyperparameters
y = rand(dist)  # Sample observations

# For Normal distribution with hyperparameters
model_normal = ExponentialFamily(Normal)
x = [0.0, 1.0] 
dist = conditional_distribution(model_normal, x; σ=0.5)
y = rand(dist)  # Sample observations
```

# Implementation
All observation models should implement this method. The default throws an error.
"""
function conditional_distribution(obs_model::ObservationModel, x; kwargs...)
    error("conditional_distribution not implemented for observation model type $(typeof(obs_model))")
end
