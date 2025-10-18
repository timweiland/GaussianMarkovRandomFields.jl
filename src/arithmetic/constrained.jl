using LinearAlgebra
using SparseArrays
using Random
using LinearSolve
using Distributions: mean, var, _rand!
import Distributions: logpdf

export ConstrainedGMRF

"""
    ConstrainedGMRF{T,L,G} <: AbstractGMRF{T,L}

A Gaussian Markov Random Field with hard linear equality constraints.

Given an unconstrained GMRF x ~ N(μ, Q⁻¹) and constraints Ax = e,
this represents the constrained distribution x | Ax = e.

This is a degenerate distribution (the precision matrix becomes singular),
but sampling and mean computation are handled efficiently using conditioning by Kriging.

# Mathematical Background

For x ~ N(μ, Q⁻¹) with constraint Ax = e, the constrained mean is:
    μ_c = μ - Q⁻¹A^T(AQ⁻¹A^T)⁻¹(Aμ - e)

And samples are obtained via:
    x_c = x - Q⁻¹A^T(AQ⁻¹A^T)⁻¹(Ax - e)

where x is a sample from the unconstrained distribution.

The constrained covariance matrix is:
    Σ_c = Q⁻¹ - Q⁻¹A^T(AQ⁻¹A^T)⁻¹AQ⁻¹

# Implementation

For efficiency, the constructor precomputes:
- Ã^T = Q⁻¹A^T (via solving QL^T = A^T where Q = LL^T)
- L_c from Cholesky factorization of AÃ^T
- B = L^(-T)Ã^T L_c^(-T) for variance computations

# Type Parameters
- `T<:Real`: The numeric type
- `L<:Union{LinearMaps.LinearMap{T}, AbstractMatrix{T}}`: The precision map type
- `G<:AbstractGMRF{T,L}`: The concrete type of the base GMRF

# Fields
- `base_gmrf::G`: The unconstrained GMRF
- `constraint_matrix::Matrix{T}`: Constraint matrix A (converted to dense)
- `constraint_vector::Vector{T}`: Constraint vector e
- `A_tilde_T::Matrix{T}`: Precomputed Q⁻¹A^T
- `L_c::Cholesky{T, Matrix{T}}`: Cholesky factorization of AÃ^T
- `constrained_mean::Vector{T}`: Precomputed constrained mean

# Constructor
    ConstrainedGMRF(base_gmrf::AbstractGMRF, A, e)

Create a constrained GMRF where `base_gmrf` is the unconstrained distribution,
`A` is the constraint matrix, and `e` is the constraint vector such that Ax = e.
"""
struct ConstrainedGMRF{T <: Real, L <: Union{LinearMaps.LinearMap{T}, AbstractMatrix{T}}, G <: AbstractGMRF{T, L}} <: AbstractGMRF{T, L}
    base_gmrf::G
    constraint_matrix::Matrix{T}
    constraint_vector::Vector{T}
    A_tilde_T::Matrix{T}
    L_c::Cholesky{T, Matrix{T}}
    constrained_mean::Vector{T}

    function ConstrainedGMRF(
            base_gmrf::G,
            A::AbstractMatrix,
            e::AbstractVector
        ) where {T <: Real, L <: Union{LinearMaps.LinearMap{T}, AbstractMatrix{T}}, G <: AbstractGMRF{T, L}}

        # Input validation
        n = length(base_gmrf)
        m, n_A = size(A)
        n == n_A || throw(ArgumentError("Constraint matrix size $(size(A)) incompatible with GMRF size $(n)"))
        m == length(e) || throw(ArgumentError("Constraint matrix rows $(m) != constraint vector length $(length(e))"))

        # Convert to appropriate types
        T_result = promote_type(T, eltype(A), eltype(e))
        A_dense = Matrix{T_result}(A)  # Convert to dense matrix for efficiency
        e_vec = Vector{T_result}(e)

        # Get the base mean and precision
        μ_base = mean(base_gmrf)
        Q = precision_map(base_gmrf)

        # Step 1: Compute Ã^T := Q⁻¹A^T
        # We solve Q * A_tilde_T = A^T for A_tilde_T
        A_T = Matrix{T_result}(A_dense')

        # For efficiency, we'll solve this using the existing LinearSolve infrastructure
        # Since we need to solve Q * X = A^T for multiple RHS columns
        A_tilde_T = Matrix{T_result}(undef, n, m)

        # Get a copy of the linsolve cache and modify it for our solve
        cache = linsolve_cache(base_gmrf)

        # Solve for each column of A^T
        for i in 1:m
            # Update the RHS in the cache
            cache.b .= A_T[:, i]
            # Solve the system
            sol = solve!(cache)
            # Store the solution
            A_tilde_T[:, i] .= sol.u
        end

        # Step 2: Compute AÃ^T and its Cholesky factorization
        AA_tilde = A_dense * A_tilde_T  # This should be m×m

        # Cholesky factorization
        L_c = cholesky(Symmetric(AA_tilde))

        # Step 3: Compute the constrained mean
        # μ_c = μ - Ã^T * L_c^(-T) * L_c^(-1) * (A*μ - e)
        # Simplifying: μ_c = μ - Ã^T * (A*A_tilde)^(-1) * (A*μ - e)
        residual = A_dense * μ_base - e_vec
        correction = A_tilde_T * (L_c \ residual)
        constrained_mean = μ_base - correction

        return new{T_result, L, G}(
            base_gmrf, A_dense, e_vec, A_tilde_T, L_c, constrained_mean
        )
    end
end

# Required AbstractGMRF interface methods
Base.length(d::ConstrainedGMRF) = length(d.base_gmrf)
mean(d::ConstrainedGMRF) = d.constrained_mean

"""
    precision_map(d::ConstrainedGMRF)

Return the precision map of the constrained GMRF.
Note: This is singular due to the constraints, but we return it for interface compliance.
In practice, this should rarely be used directly due to singularity.
"""
precision_map(d::ConstrainedGMRF) = precision_map(d.base_gmrf)  # TODO: Return constrained precision

"""
    _rand!(rng::AbstractRNG, d::ConstrainedGMRF, x::AbstractVector)

Sample from the constrained GMRF using conditioning by Kriging.

The algorithm:
1. Sample x from the unconstrained base GMRF
2. Apply Kriging correction: x_c = x - Ã^T * L_c^(-1) * (A*x - e)
"""
function _rand!(rng::AbstractRNG, d::ConstrainedGMRF{T}, x::AbstractVector{T}) where {T}
    # Step 1: Sample from unconstrained GMRF
    _rand!(rng, d.base_gmrf, x)

    # Step 2: Apply constraint correction using Kriging formula
    # x_c = x - Ã^T * L_c^(-1) * (A*x - e)
    residual = d.constraint_matrix * x - d.constraint_vector
    correction = d.A_tilde_T * (d.L_c \ residual)
    x .-= correction

    return x
end

"""
    var(d::ConstrainedGMRF)

Compute marginal variances of the constrained GMRF.

Uses the efficient formula:
    σ_c = σ - diag(B*B^T) = σ - Σⱼ B[:,j]²

where B = Ã^T * L_c^(-T) and σ is the unconstrained marginal variance.
"""
function var(d::ConstrainedGMRF{T}) where {T}
    # Get unconstrained variances
    σ_base = var(d.base_gmrf)

    B_T = d.L_c.L \ d.A_tilde_T'
    # Compute diagonal of B*B^T efficiently as row-wise sum of squares
    B_squared_rowsums = vec(sum(abs2, B_T, dims = 1))

    # Constrained variance = unconstrained variance - correction
    σ_constrained = σ_base - B_squared_rowsums

    # Ensure non-negative (numerical precision can cause tiny negative values)
    σ_constrained .= max.(σ_constrained, zero(T))

    return σ_constrained
end

function Distributions.logpdf(d::ConstrainedGMRF, z::AbstractVector)
    # Check if constraint is satisfied: A*z ≈ e
    # Points that violate the constraint have zero probability
    constraint_residual = d.constraint_matrix * z - d.constraint_vector
    if !isapprox(constraint_residual, zero(constraint_residual), atol = 1.0e-10)
        return -Inf
    end

    # Prior logpdf
    res = Distributions.logpdf(d.base_gmrf, z)

    # Constraint logpdf
    resid = d.constraint_vector - d.constraint_matrix * mean(d.base_gmrf)
    r = length(resid)
    neg_logpdf_e = 0.5 * (r * log(2π) + logdet(d.L_c) + dot(resid, d.L_c \ resid))
    res += neg_logpdf_e

    # Degenerate constraint likelihood
    # Rue and Held (2005), Section 2.3.3
    res -= 0.5 * logdet(cholesky(Symmetric(d.constraint_matrix * d.constraint_matrix')))
    return res
end

# Display methods
# COV_EXCL_START
function Base.show(io::IO, d::ConstrainedGMRF{T}) where {T}
    m, n = size(d.constraint_matrix)
    return print(io, "ConstrainedGMRF{$T}(n=$n, constraints=$m)")
end

function Base.show(io::IO, ::MIME"text/plain", d::ConstrainedGMRF{T}) where {T}
    m, n = size(d.constraint_matrix)
    println(io, "ConstrainedGMRF{$T} with $n variables and $m constraint$(m > 1 ? "s" : "")")
    println(io, "  Base GMRF: $(repr(d.base_gmrf))")
    println(io, "  Constraints: $(m)×$(n) matrix")
    return print(io, "  Constraint values: $(d.constraint_vector)")
end
# COV_EXCL_STOP
