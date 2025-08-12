using Distributions, LinearAlgebra, Random, LinearMaps
import Base: step

export joint_ssm, JointSSMMatrices

"""
    JointSSMMatrices

Abstract type for the matrices defining the transition of a certain linear
state-space model of the form

```math
G(Δt) x_{k+1} ∣ xₖ ∼ 𝒩(M(Δt) xₖ, Σ)
```

# Fields
- `Δt::Real`: Time step.
- `G::LinearMap`: Transition matrix.
- `M::LinearMap`: Observation matrix.
- `Σ⁻¹::LinearMap`: Transition precision map.
- `Σ⁻¹_sqrt::LinearMap`: Square root of the transition precision map.
- `constraint_handler`: Ferrite constraint handler.
- `constraint_noise`: Constraint noise.
"""
abstract type JointSSMMatrices end;

step(x::JointSSMMatrices) = x.Δt
get_G(x::JointSSMMatrices) = x.G
get_M(x::JointSSMMatrices) = x.M
get_Σ⁻¹(x::JointSSMMatrices) = x.Σ⁻¹
get_Σ⁻¹_sqrt(x::JointSSMMatrices) = x.Σ⁻¹_sqrt
get_constraint_handler(x::JointSSMMatrices) = x.constraint_handler
get_constraint_noise(x::JointSSMMatrices) = x.constraint_noise


@doc raw"""
    joint_ssm(x₀::GMRF, ssm_matrices::Function, ts::AbstractVector)

Form the joint GMRF for the linear state-space model given by

```math
G(Δtₖ) x_{k+1} ∣ xₖ ∼ 𝒩(M(Δtₖ) xₖ, Σ)
```

at time points given by `ts` (from which the Δtₖ are computed).
"""
joint_ssm(x₀::GMRF, ssm_matrices::Union{Function,JointSSMMatrices}, ts::AbstractVector) =
    error("joint_ssm not implemented for these argument types")

"""
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

    precision =
        SymmetricBlockTridiagonalMap(Tuple(diagonal_blocks), Tuple(off_diagonal_blocks))
    return GMRF(vcat(means...), precision)
end
"""

function joint_ssm(x₀::GMRF, ssm_mats_fn::Function, ts::AbstractRange)
    Δt = Float64(Base.step(ts))
    ssm_mats = ssm_mats_fn(Δt)
    return joint_ssm(x₀, ssm_mats, ts)
end

function joint_ssm(x₀::GMRF, ssm_mats::JointSSMMatrices, ts::AbstractRange)
    # A = G⁻¹M
    G = get_G(ssm_mats)
    G_cpy = copy(G)
    M = get_M(ssm_mats)
    Σ⁻¹ = get_Σ⁻¹(ssm_mats)
    Σ⁻¹_sqrt = get_Σ⁻¹_sqrt(ssm_mats)
    ch = get_constraint_handler(ssm_mats)
    constraint_noise = get_constraint_noise(ssm_mats)

    apply_soft_constraints!(ch, constraint_noise; K = G, Q_rhs = Σ⁻¹, Q_rhs_sqrt = Σ⁻¹_sqrt)

    means = [mean(x₀)]
    for i = 2:length(ts)
        cur_mean = M * means[i-1]
        apply_soft_constraints!(
            ch,
            constraint_noise;
            K = G_cpy,
            f_rhs = cur_mean,
            change_K = false,
        )
        push!(means, G \ cur_mean)
    end

    GᵀΣ⁻¹ = G' * Σ⁻¹
    F⁻¹ = GᵀΣ⁻¹ * G
    AᵀF⁻¹A = M' * Σ⁻¹ * M
    F⁻¹A = GᵀΣ⁻¹ * M

    Nₜ = length(ts)
    M = F⁻¹ + AᵀF⁻¹A
    diagonal_blocks = [[sparse(precision_map(x₀)) + AᵀF⁻¹A]; repeat([M], Nₜ - 2); [F⁻¹]]
    off_diagonal_blocks = repeat([-F⁻¹A], Nₜ - 1)
    diagonal_blocks = Tuple(LinearMap(block) for block in diagonal_blocks)
    off_diagonal_blocks = Tuple(LinearMap(block) for block in off_diagonal_blocks)

    precision = SymmetricBlockTridiagonalMap(diagonal_blocks, off_diagonal_blocks)
    
    # Only construct square root if x₀ has Q_sqrt available
    precision_sqrt = if x₀.Q_sqrt !== nothing
        Q_s_sqrt = LinearMap(x₀.Q_sqrt)
        F⁻¹_sqrt = LinearMap(G' * Σ⁻¹_sqrt)
        AᵀF⁻¹_sqrt = LinearMap(M' * Σ⁻¹_sqrt)
        A = hcat(Q_s_sqrt, AᵀF⁻¹_sqrt)
        B = hcat(ZeroMap{Float64}(size(Q_s_sqrt)...), -F⁻¹_sqrt)
        C = hcat(ZeroMap{Float64}(size(Q_s_sqrt)...), AᵀF⁻¹_sqrt)
        SSMBidiagonalMap(A, B, C, Nₜ)
    else
        nothing
    end

    return GMRF(vcat(means...), precision, precision_sqrt)
end
