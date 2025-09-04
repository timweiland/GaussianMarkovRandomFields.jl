using LinearAlgebra

export FixedEffectsModel

"""
    FixedEffectsModel(n::Int; λ::Real = 1e-6)

Weakly regularized fixed-effects latent component.

Encodes standard GLM-style fixed effects inside the latent vector with a small
ridge precision `λ * I(n)`.

- No hyperparameters (returns `NamedTuple()`)
- No constraints
- Name: `:fixed` (used for parameter prefixing in `CombinedModel`)
"""
struct FixedEffectsModel <: LatentModel
    n::Int
    λ::Float64

    function FixedEffectsModel(n::Int; λ::Real = 1.0e-6)
        n ≥ 0 || throw(ArgumentError("Length n must be nonnegative, got n=$n"))
        λ > 0 || throw(ArgumentError("Regularization λ must be positive, got λ=$λ"))
        return new(n, Float64(λ))
    end
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
