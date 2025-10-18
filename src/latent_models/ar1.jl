using SparseArrays
using LinearAlgebra
using LinearSolve

export AR1Model

"""
    AR1Model(n::Int; alg=LDLtFactorization(), constraint=nothing)

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
- `alg::Alg`: LinearSolve algorithm for solving linear systems
- `constraint::C`: Optional constraint, either `nothing` or `(A, e)` tuple

# Example
```julia
model = AR1Model(100)
gmrf = model(τ=2.0, ρ=0.8)  # Construct unconstrained AR1 GMRF

# With sum-to-zero constraint
model = AR1Model(100, constraint=:sumtozero)
gmrf = model(τ=2.0, ρ=0.8)  # Returns ConstrainedGMRF

# With custom constraint
A = [1.0 1.0 zeros(98)...]
e = [0.0]
model = AR1Model(100, constraint=(A, e))
gmrf = model(τ=2.0, ρ=0.8)
```
"""
struct AR1Model{Alg, C} <: LatentModel
    n::Int
    alg::Alg
    constraint::C

    function AR1Model{Alg, C}(n::Int, alg::Alg, constraint::C) where {Alg, C}
        n > 0 || throw(ArgumentError("Length n must be positive, got n=$n"))
        return new{Alg, C}(n, alg, constraint)
    end
end

function AR1Model(n::Int; alg = LDLtFactorization(), constraint = nothing)
    processed_constraint = _process_constraint(constraint, n)
    return AR1Model{typeof(alg), typeof(processed_constraint)}(n, alg, processed_constraint)
end

function Base.length(model::AR1Model)
    return model.n
end

function hyperparameters(model::AR1Model)
    return (τ = Real, ρ = Real)
end

function _validate_ar1_parameters(; τ::Real, ρ::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    abs(ρ) < 1 || throw(ArgumentError("Correlation parameter ρ must satisfy |ρ| < 1, got ρ=$ρ"))
    return nothing
end

function precision_matrix(model::AR1Model; τ::Real, ρ::Real, kwargs...)
    _validate_ar1_parameters(; τ = τ, ρ = ρ)

    n = model.n
    T = promote_type(typeof(τ), typeof(ρ))

    # Main diagonal: τ at endpoints, (1 + ρ²)τ in the middle
    main_diag = map(1:n) do i
        (i == 1 || i == n) ? T(τ) : T((1 + ρ^2) * τ)
    end

    # Off-diagonal: -ρτ for all off-diagonal elements
    off_diag = fill(-T(ρ * τ), n - 1)

    return SymTridiagonal(main_diag, off_diag)
end

function mean(model::AR1Model; kwargs...)
    return zeros(model.n)
end

function constraints(model::AR1Model; kwargs...)
    return model.constraint
end

function model_name(::AR1Model)
    return :ar1
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type
