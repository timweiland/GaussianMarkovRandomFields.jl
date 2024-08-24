using Distributions, LinearAlgebra

export joint_ssm

@doc raw"""
    joint_ssm(x₀::GMRF, AᵀF⁻¹A_fn, F⁻¹_fn, F⁻¹A_fn, ts)

Form the joint GMRF for the linear state-space model given by

```math
x_{k+1} ∣ xₖ ∼ 𝒩(A(Δtₖ) xₖ, F)
```

at time points given by `ts` (from which the Δtₖ are computed).
"""
joint_ssm(
    x₀::GMRF,
    AᵀF⁻¹A::Union{AbstractMatrix,Function},
    F⁻¹::Union{AbstractMatrix,Function},
    F⁻¹A::Union{AbstractMatrix,Function},
    ts::AbstractVector,
) = error("joint_ssm not implemented for these argument types")

function joint_ssm(
    x₀::GMRF,
    AᵀF⁻¹A::Function,
    F⁻¹::Function,
    F⁻¹A::Function,
    ts::AbstractVector,
)
    Nₛ = size(x₀.precision, 1)
    diagonal_blocks = [spzeros(size(precision_mat(x₀))) for _ in ts]
    off_diagonal_blocks = [spzeros(size(precision_mat(x₀))) for _ = 1:(length(ts)-1)]
    means = [spzeros(size(x₀)) for _ in ts]

    diagonal_blocks[1] = precision_mat(x₀)
    means[1] = mean(x₀)

    t_prev = ts[1]
    for (i, t) in enumerate(ts[2:end])
        Δt = t - t_prev
        AᵀF⁻¹A = AᵀF⁻¹A(Δt)
        F⁻¹ = F⁻¹(Δt)
        F⁻¹A = F⁻¹A(Δt)
        diagonal_blocks[i] += AᵀF⁻¹A
        off_diagonal_blocks[i] = -F⁻¹A
        diagonal_blocks[i+1] = F⁻¹
        # means[i] = A * means[i]
        t_prev = t
    end

    Nₜ = length(ts)
    global_precision = spzeros(Nₛ * Nₜ, Nₛ * Nₜ)

    for i = 1:Nₜ
        start, stop = (i - 1) * Nₛ + 1, i * Nₛ
        global_precision[start:stop, start:stop] = diagonal_blocks[i]
        if i < Nₜ
            global_precision[start:stop, stop+1:stop+Nₛ] = off_diagonal_blocks[i]'
            global_precision[stop+1:stop+Nₛ, start:stop] = off_diagonal_blocks[i]
        end
    end
    return GMRF(vcat(means...), Symmetric(global_precision))
end

function joint_ssm(
    x₀::GMRF,
    AᵀF⁻¹A_fn::Function,
    F⁻¹_fn::Function,
    F⁻¹A_fn::Function,
    ts::AbstractRange,
)
    dt = Float64(step(ts))
    AᵀF⁻¹A = AᵀF⁻¹A_fn(dt)
    F⁻¹ = F⁻¹_fn(dt)
    F⁻¹A = F⁻¹A_fn(dt)
    return joint_ssm(x₀, AᵀF⁻¹A, F⁻¹, F⁻¹A, ts)
end

function joint_ssm(
    x₀::GMRF,
    AᵀF⁻¹A::AbstractMatrix,
    F⁻¹::AbstractMatrix,
    F⁻¹A::AbstractMatrix,
    ts::AbstractRange,
)
    Nₛ = size(x₀.precision, 1)
    Nₜ = length(ts)
    M = F⁻¹ + AᵀF⁻¹A
    diagonal_blocks = [[precision_mat(x₀) + AᵀF⁻¹A]; repeat([M], Nₜ - 2); [F⁻¹]]
    off_diagonal_blocks = repeat([-F⁻¹A], Nₜ - 1)
    means = repeat([spzeros(size(x₀))], Nₜ)
    means[1] = mean(x₀)

    global_precision = spzeros(Nₛ * Nₜ, Nₛ * Nₜ)
    for i = 1:Nₜ
        start, stop = (i - 1) * Nₛ + 1, i * Nₛ
        global_precision[start:stop, start:stop] = diagonal_blocks[i]
        if i < Nₜ
            global_precision[start:stop, stop+1:stop+Nₛ] = off_diagonal_blocks[i]'
            global_precision[stop+1:stop+Nₛ, start:stop] = off_diagonal_blocks[i]
        end
    end
    return GMRF(vcat(means...), Symmetric(global_precision))
end
