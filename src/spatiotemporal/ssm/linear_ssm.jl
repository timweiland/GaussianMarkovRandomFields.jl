using Distributions, LinearAlgebra, Random, LinearMaps
import Base: step

export joint_ssm, JointSSMMatrices

abstract type JointSSMMatrices end;

step(x::JointSSMMatrices) = x.Δt
get_AᵀF⁻¹A(x::JointSSMMatrices) = x.AᵀF⁻¹A
get_F⁻¹(x::JointSSMMatrices) = x.F⁻¹
get_F⁻¹A(x::JointSSMMatrices) = x.F⁻¹A
get_F⁻¹_sqrt(x::JointSSMMatrices) = x.F⁻¹_sqrt
get_AᵀF⁻¹_sqrt(x::JointSSMMatrices) = x.AᵀF⁻¹_sqrt


@doc raw"""
    joint_ssm(x₀::GMRF, A, AᵀF⁻¹A_fn, F⁻¹_fn, F⁻¹A_fn, ts)

Form the joint GMRF for the linear state-space model given by

```math
x_{k+1} ∣ xₖ ∼ 𝒩(A(Δtₖ) xₖ, F)
```

at time points given by `ts` (from which the Δtₖ are computed).
"""
joint_ssm(x₀::GMRF, ssm_matrices::Union{Function,JointSSMMatrices}, ts::AbstractVector) =
    error("joint_ssm not implemented for these argument types")

function joint_ssm(x₀::GMRF, ssm_mats_fn::Function, ts::AbstractVector)
    Nₛ = size(x₀.precision, 1)
    diagonal_blocks = Array{LinearMap{Float64}}(undef, length(ts))
    off_diagonal_blocks = Array{LinearMap{Float64}}(undef, length(ts) - 1)
    means = [spzeros(size(x₀)) for _ in ts]

    diagonal_blocks[1] = precision_map(x₀)
    means[1] = mean(x₀)

    t_prev = ts[1]

    for (i, t) in enumerate(ts[2:end])
        Δt = t - t_prev
        ssm_mats = ssm_mats_fn(Δt)
        # TODO: Make these return linear maps by interface
        AᵀF⁻¹A = LinearMap(get_AᵀF⁻¹A(ssm_mats))
        F⁻¹ = LinearMap(get_F⁻¹(ssm_mats))
        F⁻¹A = LinearMap(get_F⁻¹A(ssm_mats))
        diagonal_blocks[i] += AᵀF⁻¹A
        off_diagonal_blocks[i] = -F⁻¹A
        diagonal_blocks[i+1] = F⁻¹
        # means[i] = A * means[i]
        t_prev = t
    end

    precision = SymmetricBlockTridiagonalMap(Tuple(diagonal_blocks), Tuple(off_diagonal_blocks))
    return GMRF(vcat(means...), precision)
end

function joint_ssm(x₀::GMRF, ssm_mats_fn::Function, ts::AbstractRange)
    Δt = Float64(Base.step(ts))
    ssm_mats = ssm_mats_fn(Δt)
    return joint_ssm(x₀, ssm_mats, ts)
end

function joint_ssm(x₀::GMRF, ssm_mats::JointSSMMatrices, ts::AbstractRange)
    AᵀF⁻¹A = get_AᵀF⁻¹A(ssm_mats)
    F⁻¹ = get_F⁻¹(ssm_mats)
    F⁻¹A = get_F⁻¹A(ssm_mats)
    F⁻¹_sqrt = get_F⁻¹_sqrt(ssm_mats)
    AᵀF⁻¹_sqrt = get_AᵀF⁻¹_sqrt(ssm_mats)

    Nₜ = length(ts)
    M = F⁻¹ + AᵀF⁻¹A
    diagonal_blocks = [[sparse(precision_map(x₀)) + AᵀF⁻¹A]; repeat([M], Nₜ - 2); [F⁻¹]]
    off_diagonal_blocks = repeat([-F⁻¹A], Nₜ - 1)
    diagonal_blocks = Tuple(LinearMap(block) for block in diagonal_blocks)
    off_diagonal_blocks = Tuple(LinearMap(block) for block in off_diagonal_blocks)
    means = repeat([spzeros(size(x₀))], Nₜ)
    means[1] = mean(x₀)

    precision = SymmetricBlockTridiagonalMap(diagonal_blocks, off_diagonal_blocks)
    Q_s_sqrt = linmap_sqrt(precision_map(x₀))
    A = hcat(Q_s_sqrt, AᵀF⁻¹_sqrt)
    B = hcat(ZeroMap{Float64}(size(Q_s_sqrt)...), -F⁻¹_sqrt)
    C = hcat(ZeroMap{Float64}(size(Q_s_sqrt)...), AᵀF⁻¹_sqrt)
    precision_sqrt = SSMBidiagonalMap(A, B, C, Nₜ)
    precision = LinearMapWithSqrt(precision, precision_sqrt)

    return GMRF(vcat(means...), precision)
end
