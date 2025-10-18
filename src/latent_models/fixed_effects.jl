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
struct FixedEffectsModel{Alg} <: LatentModel
    n::Int
    λ::Float64
    alg::Alg

    function FixedEffectsModel{Alg}(n::Int, λ::Float64, alg::Alg) where {Alg}
        n ≥ 0 || throw(ArgumentError("Length n must be nonnegative, got n=$n"))
        λ > 0 || throw(ArgumentError("Regularization λ must be positive, got λ=$λ"))
        return new{Alg}(n, λ, alg)
    end
end

function FixedEffectsModel(n::Int; λ::Real = 1.0e-6, alg = DiagonalFactorization())
    return FixedEffectsModel{typeof(alg)}(n, Float64(λ), alg)
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

function constraints(::FixedEffectsModel; kwargs...)
    return nothing
end

function model_name(::FixedEffectsModel)
    return :fixed
end
