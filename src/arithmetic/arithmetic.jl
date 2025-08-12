# LinearConditionalGMRF has been removed - conditioning now uses linear_condition function

using LinearAlgebra

export condition_on_observations, joint_gmrf, linear_condition

# Adding a deterministic vector to a GMRF
Base.:+(d::GMRF, b::AbstractVector) = GMRF(d.mean + b, d.precision)
Base.:+(b::AbstractVector, d::GMRF) = d + b
Base.:-(d::GMRF, b::AbstractVector) = GMRF(d.mean - b, d.precision)

# TODO: Find nice way to represent A * x?

""""
    joint_gmrf(
        x1::AbstractGMRF,
        A::AbstractMatrix,
        Q_ϵ::AbstractMatrix,
        b::AbstractVector=spzeros(size(A)[1])
    )

Return the joint GMRF of `x1` and `x2 = A * x1 + b + ϵ` where `ϵ ~ N(0, Q_ϵ⁻¹)`.

# Arguments
- `x1::AbstractGMRF`: The first GMRF.
- `A::AbstractMatrix`: The matrix `A`.
- `Q_ϵ::AbstractMatrix`: The precision matrix of the noise term `ϵ`.
- `b::AbstractVector=spzeros(size(A)[1])`: Offset vector `b`; optional.

# Returns
A `GMRF` object representing the joint GMRF of `x1` and `x2 = A * x1 + b + ϵ`.
"""
function joint_gmrf(
    x1::AbstractGMRF,
    A::AbstractMatrix,
    Q_ϵ::AbstractMatrix,
    b::AbstractVector = spzeros(size(A)[1]),
)
    # TODO: Think about using LinearMap implementation here
    x1_mean, x1_precision = mean(x1), sparse(precision_map(x1))
    off_diagonal = -Q_ϵ * A
    Q_joint = [
        x1_precision+A'*Q_ϵ*A off_diagonal'
        off_diagonal Q_ϵ
    ]
    x2_mean = A * x1_mean
    if b !== nothing
        x2_mean += b
    end
    μ_joint = [x1_mean; x2_mean]
    return GMRF(μ_joint, Symmetric(Q_joint))
end

""""
    condition_on_observations(
        x::GMRF,
        A::Union{AbstractMatrix,LinearMap},
        Q_ϵ::Union{AbstractMatrix,LinearMap,Real},
        y::AbstractVector=zeros(size(A)[1]),
        b::AbstractVector=zeros(size(A)[1]);
        # solver_blueprint parameter removed - no longer needed with LinearSolve
    )

Condition a GMRF `x` on observations `y = A * x + b + ϵ` where `ϵ ~ N(0, Q_ϵ⁻¹)`.

# Arguments
- `x::GMRF`: The GMRF to condition on.
- `A::Union{AbstractMatrix,LinearMap}`: The matrix `A`.
- `Q_ϵ::Union{AbstractMatrix,LinearMap, Real}`: The precision matrix of the
         noise term `ϵ`. In case a real number is provided, it is interpreted
         as a scalar multiple of the identity matrix.
- `y::AbstractVector=zeros(size(A)[1])`: The observations `y`; optional.
- `b::AbstractVector=zeros(size(A)[1])`: Offset vector `b`; optional.

# Keyword arguments
# Note: solver_blueprint parameter removed - no longer needed with LinearSolve.jl

# Returns
A `GMRF` object representing the conditional GMRF `x | (y = A * x + b + ϵ)`.

# Notes
This function now delegates to `linear_condition` for improved efficiency.
"""
function condition_on_observations(
    x::GMRF,
    A::Union{AbstractMatrix,LinearMap},
    Q_ϵ::Union{AbstractMatrix,LinearMap,Real},
    y::AbstractVector = zeros(size(A, 1)),
    b::AbstractVector = zeros(size(A, 1));
    # solver_blueprint parameter removed - no longer needed with LinearSolve
)
    # Convert scalar Q_ϵ to UniformScalingMap for compatibility
    if Q_ϵ isa Real
        Q_ϵ = LinearMaps.UniformScalingMap(Q_ϵ, size(A, 1))
    end
    # Delegate to new linear_condition function
    return linear_condition(x; A=A, Q_ϵ=Q_ϵ, y=y, b=b)
end

"""
    linear_condition(gmrf::GMRF; A, Q_ϵ, y, b=zeros(size(A, 1)))

Condition a GMRF on linear observations y = A * x + b + ϵ where ϵ ~ N(0, Q_ϵ^(-1)).

# Arguments
- `gmrf::GMRF`: The prior GMRF
- `A::Union{AbstractMatrix, LinearMap}`: Observation matrix
- `Q_ϵ::Union{AbstractMatrix, LinearMap}`: Precision matrix of observation noise
- `y::AbstractVector`: Observation values
- `b::AbstractVector`: Offset vector (defaults to zeros)

# Returns
A new `GMRF` representing the posterior distribution with updated mean and precision.

# Notes
This replaces the deprecated `LinearConditionalGMRF` type with a functional approach.
Uses information vector arithmetic for efficient conditioning without intermediate solves.
"""
function linear_condition(gmrf::GMRF; A, Q_ϵ, y, b=zeros(size(A, 1)))
    # Ensure everything is compatible types
    A = A isa LinearMap ? A : LinearMap(A)
    if Q_ϵ isa Real
        Q_ϵ = LinearMaps.UniformScalingMap(Q_ϵ, size(A, 1))
    elseif !(Q_ϵ isa LinearMap)
        Q_ϵ = LinearMap(Q_ϵ)
    end

    # Compute posterior precision: Q_posterior = Q_prior + A' * Q_ϵ * A
    Q_posterior = precision_map(gmrf) + A' * Q_ϵ * A
    
    # Update information: info_posterior = info_prior + A' * Q_ϵ * (y - b)
    info_posterior = information_vector(gmrf) + A' * (Q_ϵ * (y - b))
    
    # Create new GMRF from information vector
    # TODO: Compute Q_sqrt for conditioned GMRF - non-trivial, skipping for now
    return GMRF(InformationVector(info_posterior), Q_posterior, gmrf.linsolve_cache.alg; Q_sqrt=nothing)
end

# MetaGMRF conditioning - preserves wrapper type and metadata
function linear_condition(mgmrf::MetaGMRF; kwargs...)
    conditioned_gmrf = linear_condition(mgmrf.gmrf; kwargs...)
    return MetaGMRF(conditioned_gmrf, mgmrf.metadata)
end

function condition_on_observations(mgmrf::MetaGMRF, args...; kwargs...)
    conditioned_gmrf = condition_on_observations(mgmrf.gmrf, args...; kwargs...)
    return MetaGMRF(conditioned_gmrf, mgmrf.metadata)
end
