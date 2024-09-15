using LinearAlgebra

export condition_on_observations, joint_gmrf

# Adding a deterministic vector to a GMRF
Base.:+(d::GMRF, b::AbstractVector) = GMRF(d.mean + b, d.precision)
Base.:+(b::AbstractVector, d::GMRF) = d + b
Base.:-(d::GMRF, b::AbstractVector) = GMRF(d.mean - b, d.precision)

# TODO: Find nice way to represent A * x?

""""
    joint_gmrf(x1::GMRF, A::AbstractMatrix, Q_ϵ::AbstractMatrix,
    b::AbstractVector=spzeros(size(A)[1])

Return the joint GMRF of `x1` and `x2 = A * x1 + b + ϵ` where `ϵ ~ N(0, Q_ϵ)`.

`b` is optional and defaults to a zero vector.
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
    condition_on_observations(x::GMRF, A::AbstractMatrix, Q_ϵ::AbstractMatrix,
    y::AbstractVector=spzeros(size(A)[1]), b::AbstractVector=spzeros(size(A)[1])

Condition a GMRF `x` on observations `y = A * x + b + ϵ` where `ϵ ~ N(0, Q_ϵ)`.

`y` and `b` are optional and default to zero vectors.
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
