using SparseArrays
using LinearAlgebra

export IIDModel

"""
    IIDModel(n::Int)

An independent and identically distributed (IID) latent model for constructing simple diagonal GMRFs.

The IID model represents independent Gaussian random variables with identical precision τ.
This is the simplest possible latent model, equivalent to a scaled identity precision matrix.

# Mathematical Description

Each element is independent: x[i] ~ N(0, τ⁻¹) for i = 1,...,n.
The precision matrix is simply: Q = τ * I(n)

This model is useful for:
- Modeling independent effects or noise
- Baseline comparisons with structured models
- Teaching/demonstration purposes

# Hyperparameters
- `τ`: Precision parameter (τ > 0)

# Fields  
- `n::Int`: Length of the IID process

# Example
```julia
model = IIDModel(100)
gmrf = model(τ=2.0)  # Returns GMRF with precision 2.0 * I(100)
```
"""
struct IIDModel <: LatentModel
    n::Int

    function IIDModel(n::Int)
        n > 0 || throw(ArgumentError("Length n must be positive, got n=$n"))
        return new(n)
    end
end

function Base.length(model::IIDModel)
    return model.n
end

function hyperparameters(model::IIDModel)
    return (τ = Real,)
end

function _validate_iid_parameters(; τ::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    return nothing
end

function precision_matrix(model::IIDModel; τ::Real, kwargs...)
    _validate_iid_parameters(; τ = τ)

    n = model.n
    T = typeof(τ)

    # Simple diagonal matrix: τ * I
    return Diagonal(fill(T(τ), n))
end

function mean(model::IIDModel; kwargs...)
    return zeros(model.n)
end

function constraints(model::IIDModel; kwargs...)
    return nothing  # IID has no constraints
end

function model_name(::IIDModel)
    return :iid
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type
