using LinearAlgebra, SparseArrays

export joint_gmrf

""""
    joint_gmrf(
        x1::AbstractGMRF,
        A::AbstractMatrix,
        Q_ϵ::AbstractMatrix,
        b::AbstractVector=spzeros(size(A, 1))
    )

Return the joint GMRF of `x1` and `x2 = A * x1 + b + ϵ` where `ϵ ~ N(0, Q_ϵ⁻¹)`.

# Arguments
- `x1::AbstractGMRF`: The first GMRF.
- `A::AbstractMatrix`: The matrix `A`.
- `Q_ϵ::AbstractMatrix`: The precision matrix of the noise term `ϵ`.
- `b::AbstractVector=spzeros(size(A, 1))`: Offset vector `b`; optional.

# Returns
A `GMRF` object representing the joint GMRF of `x1` and `x2 = A * x1 + b + ϵ`.
"""
function joint_gmrf(
        x1::AbstractGMRF,
        A::AbstractMatrix,
        Q_ϵ::AbstractMatrix,
        b::AbstractVector = spzeros(size(A, 1)::Int),
    )
    # TODO: Think about using LinearMap implementation here
    x1_mean, x1_precision = mean(x1), sparse(precision_map(x1))
    off_diagonal = -Q_ϵ * A
    Q_joint = [
        x1_precision + A' * Q_ϵ * A off_diagonal'
        off_diagonal Q_ϵ
    ]
    x2_mean = A * x1_mean + b
    μ_joint = [x1_mean; x2_mean]
    return GMRF(μ_joint, Symmetric(Q_joint))
end
