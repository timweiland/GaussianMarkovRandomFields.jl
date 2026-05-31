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

# ---------------------------------------------------------------------------
# Shared helpers for observation models whose components (e.g. a design matrix
# or a residual function) may optionally depend on hyperparameters θ. These
# back the "resolve θ at the materialization seam" pattern: a model declares
# the extra θ-names a parameterized component consumes, merges them into its
# `hyperparameters`, and projects the materialization kwargs onto them.
# ---------------------------------------------------------------------------

# Ordered union of two hyperparameter-name tuples: entries of `a` first, then
# any of `b` not already present. The empty-`b` method returns `a` unchanged
# (no allocation) so the fixed, non-parameterized path is a no-op.
_merge_hyperparameter_names(a::Tuple{Vararg{Symbol}}, ::Tuple{}) = a
function _merge_hyperparameter_names(a::Tuple{Vararg{Symbol}}, b::Tuple{Vararg{Symbol}})
    out = collect(Symbol, a)
    for s in b
        s in out || push!(out, s)
    end
    return Tuple(out)
end

# Project the materialization kwargs `θ` onto the `names` a parameterized
# component declared, erroring with a clear message if any are missing. The
# empty-`names` method returns the empty NamedTuple with no work, keeping the
# fixed path allocation-free.
_project_hyperparameters(::Tuple{}, ::NamedTuple) = NamedTuple()
function _project_hyperparameters(names::Tuple{Vararg{Symbol}}, θ::NamedTuple)
    missing_names = filter(n -> !haskey(θ, n), names)
    isempty(missing_names) || throw(
        ArgumentError(
            "Missing hyperparameter(s) $(missing_names) required by a parameterized " *
                "observation-model component. Supplied keys: $(keys(θ))."
        )
    )
    return NamedTuple{names}(θ)
end

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
# COV_EXCL_START
function conditional_distribution(obs_model::ObservationModel, x; kwargs...)
    throw(MethodError(conditional_distribution, (obs_model, x)))
end
# COV_EXCL_STOP
