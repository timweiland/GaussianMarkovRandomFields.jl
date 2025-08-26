using SparseArrays
using LinearAlgebra

export AR1Model

"""
    AR1Model(n::Int)

A first-order autoregressive (AR1) latent model for constructing AR1 GMRFs.

The AR1 model represents a temporal process where each observation depends on 
the previous observation with some correlation ρ and precision τ.

# Mathematical Description

For n observations, the AR1 process has:
- Zero mean: μ = 0
- Precision matrix Q with tridiagonal structure:
  - Q[1,1] = τ
  - Q[i,i] = (1 + ρ²)τ for i = 2,...,n-1  
  - Q[n,n] = τ
  - Q[i,i+1] = Q[i+1,i] = -ρτ for i = 1,...,n-1

# Hyperparameters
- `τ`: Precision parameter (τ > 0)
- `ρ`: Correlation parameter (|ρ| < 1)

# Fields
- `n::Int`: Length of the AR1 process

# Example
```julia
model = AR1Model(100)
gmrf = model(τ=2.0, ρ=0.8)  # Construct AR1 GMRF
```
"""
struct AR1Model <: LatentModel
    n::Int

    function AR1Model(n::Int)
        n > 0 || throw(ArgumentError("Length n must be positive, got n=$n"))
        return new(n)
    end
end

function hyperparameters(model::AR1Model)
    return (τ = Real, ρ = Real)
end

function _validate_ar1_parameters(; τ::Real, ρ::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    abs(ρ) < 1 || throw(ArgumentError("Correlation parameter ρ must satisfy |ρ| < 1, got ρ=$ρ"))
    return nothing
end

function precision_matrix(model::AR1Model; τ::Real, ρ::Real)
    _validate_ar1_parameters(; τ = τ, ρ = ρ)

    n = model.n
    T = promote_type(typeof(τ), typeof(ρ))

    if n == 1
        # Single element case
        main_diag = [T(τ)]
        off_diag = T[]
    else
        # Main diagonal: τ at endpoints, (1 + ρ²)τ in the middle
        main_diag = Vector{T}(undef, n)
        main_diag[1] = τ
        main_diag[end] = τ
        for i in 2:(n - 1)
            main_diag[i] = (1 + ρ^2) * τ
        end

        # Off-diagonal: -ρτ for all off-diagonal elements
        off_diag = fill(-ρ * τ, n - 1)
    end

    return SymTridiagonal(main_diag, off_diag)
end

function mean(model::AR1Model; kwargs...)
    return zeros(model.n)
end

function constraints(model::AR1Model; kwargs...)
    return nothing  # AR1 has no constraints
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type
