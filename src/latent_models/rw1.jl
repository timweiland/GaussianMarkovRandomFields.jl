using SparseArrays
using LinearAlgebra

export RW1Model

"""
    RW1Model(n::Int)

A first-order random walk (RW1) latent model for constructing intrinsic GMRFs.

The RW1 model represents a non-stationary temporal process where each observation 
is the previous observation plus Gaussian noise. This creates a smooth trend model
that's popular for temporal smoothing and time-varying effects.

# Mathematical Description

The RW1 process defines increments: x[i+1] - x[i] ~ N(0, τ⁻¹) for i = 1,...,n-1.
This leads to a singular precision matrix with the tridiagonal structure:
- Q[1,1] = 1, Q[n,n] = 1  
- Q[i,i] = 2 for i = 2,...,n-1
- Q[i,i+1] = Q[i+1,i] = -1 for i = 1,...,n-1

Since this matrix is singular (rank n-1), we handle it as an intrinsic GMRF by:
1. Scaling by τ first, then adding small regularization (1e-5) to diagonal for numerical stability
2. Adding sum-to-zero constraint: sum(x) = 0

# Hyperparameters
- `τ`: Precision parameter (τ > 0)

# Fields  
- `n::Int`: Length of the RW1 process
- `regularization::Float64`: Small value added to diagonal after scaling (default 1e-5)

# Example
```julia
model = RW1Model(100)
gmrf = model(τ=1.0)  # Returns ConstrainedGMRF with sum-to-zero constraint
```
"""
struct RW1Model <: LatentModel
    n::Int
    regularization::Float64

    function RW1Model(n::Int; regularization::Float64 = 1.0e-5)
        n > 1 || throw(ArgumentError("RW1 requires length n > 1, got n=$n"))
        regularization > 0 || throw(ArgumentError("Regularization must be positive, got $regularization"))
        return new(n, regularization)
    end
end

function Base.length(model::RW1Model)
    return model.n
end

function hyperparameters(model::RW1Model)
    return (τ = Real,)
end

function _validate_rw1_parameters(; τ::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    return nothing
end

function precision_matrix(model::RW1Model; τ::Real, kwargs...)
    _validate_rw1_parameters(; τ = τ)

    n = model.n
    T = promote_type(typeof(τ), Float64)  # Ensure Float64 for regularization

    # Build the main diagonal: 1 at endpoints, 2 in the middle
    # Scale by τ and add regularization (using map to avoid Zygote mutation issues)
    main_diag = map(1:n) do i
        base_val = (i == 1 || i == n) ? T(1) : T(2)
        base_val * τ + model.regularization
    end

    # Off-diagonal: -1 (scaled by τ)
    off_diag = fill(-T(τ), n - 1)

    return SymTridiagonal(main_diag, off_diag)
end

function mean(model::RW1Model; kwargs...)
    return zeros(model.n)
end

function constraints(model::RW1Model; kwargs...)
    # Sum-to-zero constraint: sum(x) = 0
    # A is 1×n matrix of all ones, e is [0]
    n = model.n
    A = ones(1, n)  # 1×n matrix
    e = [0.0]       # Constraint vector
    return (A, e)
end

function model_name(::RW1Model)
    return :rw1
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type
