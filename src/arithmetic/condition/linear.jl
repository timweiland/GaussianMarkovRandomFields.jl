using LinearMaps

export linear_condition
# DEPRECATED:
export condition_on_observations

"""
    prepare_map(Q_prior, x) -> x_converted

Helper function to ensure x is in the same format as Q_prior.
Uses multiple dispatch to handle AbstractMatrix vs LinearMap cases.
"""
# When Q_prior is a matrix, convert to matrix
prepare_map(::AbstractMatrix, M::AbstractMatrix, ::Int) = M
prepare_map(::AbstractMatrix, M::UniformScaling, ::Int) = M
prepare_map(::AbstractMatrix, x::Real, ::Int) = x * I
prepare_map(::AbstractMatrix, M::LinearMap, ::Int) = to_matrix(M)

# When Q_prior is a LinearMap, convert to LinearMap
prepare_map(::LinearMap, M::LinearMap, ::Int) = M
prepare_map(::LinearMap, M::AbstractMatrix, ::Int) = LinearMap(M)
prepare_map(::LinearMap, M::UniformScaling, n::Int) = LinearMaps.UniformScalingMap(M.λ, n)
prepare_map(::LinearMap, x::Real, n::Int) = LinearMaps.UniformScalingMap(x, n)

"""
    linear_condition(gmrf::GMRF; A, Q_ϵ, y, b=zeros(size(A, 1)), obs_precision_contrib=nothing)

Condition a GMRF on linear observations y = A * x + b + ϵ where ϵ ~ N(0, Q_ϵ^(-1)).

# Arguments
- `gmrf::GMRF`: The prior GMRF
- `A::Union{AbstractMatrix, LinearMap}`: Observation matrix
- `Q_ϵ::Union{AbstractMatrix, LinearMap}`: Precision matrix of observation noise
- `y::AbstractVector`: Observation values
- `b::AbstractVector`: Offset vector (defaults to zeros)
- `obs_precision_contrib`: Precomputed A' * Q_ϵ * A (optional optimization)

# Returns
A new `GMRF` representing the posterior distribution with updated mean and precision.

# Performance Optimization
When A has special structure (e.g., index selection), users can precompute
A' * Q_ϵ * A more efficiently and pass it to avoid redundant computation.

# Notes
Uses information vector arithmetic for efficient conditioning without intermediate solves.
"""
function linear_condition(gmrf::GMRF; A, Q_ϵ, y, b = zeros(size(A, 1)), obs_precision_contrib = nothing)
    # Prepare common components
    Q_prior = precision_map(gmrf)
    A = prepare_map(Q_prior, A, size(Q_prior, 1))
    Q_ϵ = prepare_map(Q_prior, Q_ϵ, A isa UniformScaling ? size(Q_prior, 1) : size(A, 1))

    # Compute or use precomputed A' * Q_ϵ * A
    if obs_precision_contrib === nothing
        obs_precision_contrib = A' * Q_ϵ * A
    else
        obs_precision_contrib = prepare_map(Q_prior, obs_precision_contrib, size(Q_prior, 1))
    end

    # Compute posterior precision and information
    Q_posterior = Q_prior + obs_precision_contrib
    info_posterior = information_vector(gmrf) + A' * (Q_ϵ * (y - b))

    return GMRF(InformationVector(info_posterior), Q_posterior, gmrf.linsolve_cache.alg; Q_sqrt = nothing)
end

# MetaGMRF conditioning - preserves wrapper type and metadata
function linear_condition(mgmrf::MetaGMRF; kwargs...)
    conditioned_gmrf = linear_condition(mgmrf.gmrf; kwargs...)
    return MetaGMRF(conditioned_gmrf, mgmrf.metadata)
end

####################
##   DEPRECATED   ##
####################
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
This function is deprecated. Use `linear_condition`.
"""
function condition_on_observations(
        x::GMRF,
        A::Union{AbstractMatrix, LinearMap},
        Q_ϵ::Union{AbstractMatrix, LinearMap, Real},
        y::AbstractVector = zeros(size(A, 1)),
        b::AbstractVector = zeros(size(A, 1))
        # solver_blueprint parameter removed - no longer needed with LinearSolve
    )
    # Delegate to new linear_condition function
    return linear_condition(x; A = A, Q_ϵ = Q_ϵ, y = y, b = b)
end

function condition_on_observations(mgmrf::MetaGMRF, args...; kwargs...)
    conditioned_gmrf = condition_on_observations(mgmrf.gmrf, args...; kwargs...)
    return MetaGMRF(conditioned_gmrf, mgmrf.metadata)
end

# ConstrainedGMRF linear conditioning - apply conditioning to base GMRF then re-constrain
function linear_condition(constrained_gmrf::ConstrainedGMRF; kwargs...)
    # Apply linear conditioning to the base GMRF
    conditioned_base = linear_condition(constrained_gmrf.base_gmrf; kwargs...)

    # Re-apply the same constraints to the conditioned GMRF
    return ConstrainedGMRF(
        conditioned_base,
        constrained_gmrf.constraint_matrix,
        constrained_gmrf.constraint_vector
    )
end
