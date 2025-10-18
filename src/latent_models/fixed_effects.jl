using LinearAlgebra
using LinearSolve

export FixedEffectsModel

"""
    FixedEffectsModel(n::Int; λ::Real = 1e-6, alg=DiagonalFactorization())

Weakly regularized fixed-effects latent component.

Encodes standard GLM-style fixed effects inside the latent vector with a small
ridge precision `λ * I(n)`.

- No hyperparameters (returns `NamedTuple()`)
- No constraints
- Name: `:fixed` (used for parameter prefixing in `CombinedModel`)

# Fields
- `n::Int`: Length of the fixed effects vector
- `λ::Float64`: Ridge regularization parameter
- `alg::Alg`: LinearSolve algorithm for solving linear systems

# Example
```julia
model = FixedEffectsModel(10)
gmrf = model()  # Returns GMRF with precision λ * I(10) using DiagonalFactorization

# Or specify custom algorithm
model = FixedEffectsModel(10, alg=CHOLMODFactorization())
gmrf = model()
```
"""
struct FixedEffectsModel{Alg, C} <: LatentModel
    n::Int
    λ::Float64
    alg::Alg
    constraint::C

    function FixedEffectsModel{Alg, C}(n::Int, λ::Float64, alg::Alg, constraint::C) where {Alg, C}
        n ≥ 0 || throw(ArgumentError("Length n must be nonnegative, got n=$n"))
        λ > 0 || throw(ArgumentError("Regularization λ must be positive, got λ=$λ"))
        return new{Alg, C}(n, λ, alg, constraint)
    end
end

function FixedEffectsModel(n::Int; λ::Real = 1.0e-6, alg = DiagonalFactorization(), constraint = nothing)
    processed_constraint = _process_constraint(constraint, n)
    return FixedEffectsModel{typeof(alg), typeof(processed_constraint)}(n, Float64(λ), alg, processed_constraint)
end

function Base.length(model::FixedEffectsModel)
    return model.n
end

function hyperparameters(::FixedEffectsModel)
    return NamedTuple()
end

function precision_matrix(model::FixedEffectsModel; kwargs...)
    return model.λ * I(model.n)
end

function mean(model::FixedEffectsModel; kwargs...)
    return zeros(model.n)
end

function constraints(model::FixedEffectsModel; kwargs...)
    return model.constraint
end

function model_name(::FixedEffectsModel)
    return :fixed
end
