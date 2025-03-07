include("linear_conditional_gmrf.jl")

using LinearAlgebra

export condition_on_observations, joint_gmrf

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
        x::AbstractGMRF,
        A::Union{AbstractMatrix,LinearMap},
        Q_ϵ::Union{AbstractMatrix,LinearMap,Real},
        y::AbstractVector=spzeros(size(A)[1]),
        b::AbstractVector=spzeros(size(A)[1]);
        solver_blueprint::AbstractSolverBlueprint=CholeskySolverBlueprint()
    )

Condition a GMRF `x` on observations `y = A * x + b + ϵ` where `ϵ ~ N(0, Q_ϵ⁻¹)`.

# Arguments
- `x::AbstractGMRF`: The GMRF to condition on.
- `A::Union{AbstractMatrix,LinearMap}`: The matrix `A`.
- `Q_ϵ::Union{AbstractMatrix,LinearMap, Real}`: The precision matrix of the
         noise term `ϵ`. In case a real number is provided, it is interpreted
         as a scalar multiple of the identity matrix.
- `y::AbstractVector=spzeros(size(A)[1])`: The observations `y`; optional.
- `b::AbstractVector=spzeros(size(A)[1])`: Offset vector `b`; optional.

# Keyword arguments
- `solver_blueprint::AbstractSolverBlueprint=CholeskySolverBlueprint()`:
         The solver blueprint; optional.

# Returns
A `LinearConditionalGMRF` object representing
the conditional GMRF `x | (y = A * x + b + ϵ)`.
"""
function condition_on_observations(
    x::AbstractGMRF,
    A::Union{AbstractMatrix,LinearMap},
    Q_ϵ::Union{AbstractMatrix,LinearMap,Real},
    y::AbstractVector = spzeros(Base.size(A)[1]),
    b::AbstractVector = spzeros(Base.size(A)[1]);
    solver_blueprint::AbstractSolverBlueprint = CholeskySolverBlueprint(),
)
    if Q_ϵ isa Real
        Q_ϵ = LinearMaps.UniformScalingMap(Q_ϵ, Base.size(A)[1])
    end
    return LinearConditionalGMRF(x, A, Q_ϵ, y, b, solver_blueprint)
end
