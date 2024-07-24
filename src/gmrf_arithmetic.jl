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
    x1::GMRF,
    A::AbstractMatrix,
    Q_ϵ::AbstractMatrix,
    b::AbstractVector = spzeros(size(A)[1]),
)
    off_diagonal = -Q_ϵ * A
    Q_joint = [
        x1.precision+A'*Q_ϵ*A off_diagonal'
        off_diagonal Q_ϵ
    ]
    x2_mean = A * x1.mean
    if b !== nothing
        x2_mean += b
    end
    μ_joint = [x1.mean; x2_mean]
    return GMRF(μ_joint, Q_joint)
end

""""
    condition_on_observations(x::GMRF, A::AbstractMatrix, Q_ϵ::AbstractMatrix,
    y::AbstractVector=spzeros(size(A)[1]), b::AbstractVector=spzeros(size(A)[1])

Condition a GMRF `x` on observations `y = A * x + b + ϵ` where `ϵ ~ N(0, Q_ϵ)`.

`y` and `b` are optional and default to zero vectors.
"""
function condition_on_observations(
    x::GMRF,
    A::AbstractMatrix,
    Q_ϵ::AbstractMatrix,
    y::AbstractVector = spzeros(size(A)[1]),
    b::AbstractVector = spzeros(size(A)[1]),
)
    Q_cond = x.precision + A' * Q_ϵ * A
    Q_cond_chol = cholesky(Hermitian(Q_cond))
    residual = y - (A * x.mean + b)
    μ_cond = x.mean + Q_cond_chol \ (A' * (Q_ϵ * residual))
    return GMRF(μ_cond, Q_cond, Q_cond_chol)
end
