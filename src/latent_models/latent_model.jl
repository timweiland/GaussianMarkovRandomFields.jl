export LatentModel, hyperparameters, precision_matrix, mean, constraints, model_name

"""
    LatentModel

Abstract type for latent variable models that can be used to construct GMRFs.

A LatentModel provides a structured way to define commonly used GMRFs such as 
AR1, RW1, and other temporal/spatial models by specifying:

1. The hyperparameters of the model
2. How to construct the precision matrix from hyperparameters  
3. How to construct the mean vector from hyperparameters
4. Any linear constraints that should be applied

# Interface

Each concrete subtype must implement:
- `length(model)`: Return the size/dimension of the latent process
- `hyperparameters(model)`: Return a NamedTuple describing the hyperparameters
- `precision_matrix(model; kwargs...)`: Construct precision matrix from hyperparameter values
- `mean(model; kwargs...)`: Construct mean vector from hyperparameter values  
- `constraints(model; kwargs...)`: Return constraint information or `nothing`
- `model_name(model)`: Return a Symbol representing the preferred name for this model type
- `(model)(; kwargs...)`: Instantiate a concrete GMRF from hyperparameter values

# Usage

```julia
# Define a model
model = SomeLatentModel(n=100)

# Get hyperparameter specification
params = hyperparameters(model)  # e.g. (τ=Real, ρ=Real)

# Instantiate GMRF with specific parameter values
gmrf = model(τ=2.0, ρ=0.8)  # Returns GMRF or ConstrainedGMRF
```
"""
abstract type LatentModel end

"""
    Base.length(model::LatentModel)

Return the size/dimension of the latent process.

# Returns
An integer representing the number of latent variables in the model.
"""
function Base.length(model::LatentModel)
    error("length not implemented for $(typeof(model))")
end

"""
    hyperparameters(model::LatentModel)

Return a NamedTuple describing the hyperparameters and their types for the model.

# Returns
A NamedTuple where keys are parameter names and values are their expected types.
"""
function hyperparameters(model::LatentModel)
    error("hyperparameters not implemented for $(typeof(model))")
end

"""
    precision_matrix(model::LatentModel; kwargs...)

Construct the precision matrix for the model given hyperparameter values.

# Arguments  
- `model`: The LatentModel instance
- `kwargs...`: Hyperparameter values as keyword arguments

# Returns
A precision matrix (AbstractMatrix or LinearMap) for use in GMRF construction.
"""
function precision_matrix(model::LatentModel; kwargs...)
    error("precision_matrix not implemented for $(typeof(model))")
end

"""
    mean(model::LatentModel; kwargs...)

Construct the mean vector for the model given hyperparameter values.

# Arguments
- `model`: The LatentModel instance  
- `kwargs...`: Hyperparameter values as keyword arguments

# Returns
A mean vector (AbstractVector) for use in GMRF construction.
"""
function mean(model::LatentModel; kwargs...)
    error("mean not implemented for $(typeof(model))")
end

"""
    constraints(model::LatentModel; kwargs...)

Return constraint information for the model given hyperparameter values.

# Arguments
- `model`: The LatentModel instance
- `kwargs...`: Hyperparameter values as keyword arguments  

# Returns
Either `nothing` if no constraints, or a tuple `(A, e)` where `A` is the 
constraint matrix and `e` is the constraint vector such that `Ax = e`.
"""
function constraints(model::LatentModel; kwargs...)
    error("constraints not implemented for $(typeof(model))")
end

"""
    model_name(model::LatentModel)

Return a Symbol representing the preferred name for this model type.

This name is used for parameter prefixing in CombinedModel to avoid conflicts.
For example, if two models both have a τ parameter, they become τ_ar1, τ_besag, etc.

# Returns
A Symbol that will be used as the suffix in parameter names.
"""
function model_name(model::LatentModel)
    error("model_name not implemented for $(typeof(model))")
end

"""
    (model::LatentModel)(; kwargs...)

Instantiate a concrete GMRF from the LatentModel with given hyperparameter values.

This method constructs the appropriate GMRF type:
- `GMRF` if `constraints(model; kwargs...)` returns `nothing`
- `ConstrainedGMRF` if `constraints(model; kwargs...)` returns constraint information

# Arguments
- `kwargs...`: Hyperparameter values as keyword arguments

# Returns
An `AbstractGMRF` instance (either `GMRF` or `ConstrainedGMRF`).
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
