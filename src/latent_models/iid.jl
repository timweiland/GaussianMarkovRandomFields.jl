using SparseArrays
using LinearAlgebra
using LinearSolve

export IIDModel

"""
    IIDModel(n::Int; alg=DiagonalFactorization(), constraint=nothing)

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
- `alg::Alg`: LinearSolve algorithm for solving linear systems
- `constraint::C`: Optional constraint, either `nothing` or `(A, e)` tuple

# Example
```julia
model = IIDModel(100)
gmrf = model(τ=2.0)  # Returns unconstrained GMRF

# With sum-to-zero constraint (common in INLA for separating global offset)
model = IIDModel(100, constraint=:sumtozero)
gmrf = model(τ=2.0)  # Returns ConstrainedGMRF with sum-to-zero constraint
```
"""
struct IIDModel{Alg, C} <: LatentModel
    n::Int
    alg::Alg
    constraint::C

    function IIDModel{Alg, C}(n::Int, alg::Alg, constraint::C) where {Alg, C}
        n > 0 || throw(ArgumentError("Length n must be positive, got n=$n"))
        return new{Alg, C}(n, alg, constraint)
    end
end

function IIDModel(n::Int; alg = DiagonalFactorization(), constraint = nothing)
    processed_constraint = _process_constraint(constraint, n)
    return IIDModel{typeof(alg), typeof(processed_constraint)}(n, alg, processed_constraint)
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
    return model.constraint
end

function model_name(::IIDModel)
    return :iid
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type
