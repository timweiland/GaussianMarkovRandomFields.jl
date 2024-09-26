using SparseArrays

export ImplicitEulerSSM, ImplicitEulerJointSSMMatrices, joint_ssm

"""
    ImplicitEulerSSM(x₀, G, A, E, spatial_noise)

State-space model for the implicit Euler discretization of a stochastic
differential equation.

The state-space model is given by

```
G(Δt) xₖ = M(Δt) xₖ + M(Δt) β(Δt) zₛ
```

where `zₛ` is (possibly colored) spatial noise. 
"""
struct ImplicitEulerSSM
    x₀::AbstractGMRF
    G::Function
    M::Function
    M⁻¹::Function
    β::Function
    β⁻¹::Function
    spatial_noise::AbstractGMRF
    ts::AbstractVector

    function ImplicitEulerSSM(x₀, G, M, M⁻¹, β, β⁻¹, spatial_noise, ts::AbstractVector)
        new(x₀, G, M, M⁻¹, β, β⁻¹, spatial_noise, ts::AbstractVector)
    end
end

struct ImplicitEulerJointSSMMatrices <: JointSSMMatrices
    Δt::Real
    AᵀF⁻¹A::AbstractMatrix
    F⁻¹::AbstractMatrix
    F⁻¹A::AbstractMatrix

    function ImplicitEulerJointSSMMatrices(ssm::ImplicitEulerSSM, Δt::Real)
        G = sparse(ssm.G(Δt))
        M⁻¹ = sparse(ssm.M⁻¹(Δt))
        β⁻¹ = ssm.β⁻¹(Δt)
        # β⁻² = β⁻¹^2
        Q_s = to_matrix(precision_map(ssm.spatial_noise))

        AᵀF⁻¹A = sparse(β⁻¹') * (Q_s * β⁻¹)
        M⁻¹G = M⁻¹ * G
        F⁻¹A = sparse(M⁻¹G') * AᵀF⁻¹A
        F⁻¹ = F⁻¹A * M⁻¹G

        return new(Δt, AᵀF⁻¹A, F⁻¹, F⁻¹A)
    end
end

function joint_ssm(ssm::ImplicitEulerSSM)
    x₀ = ssm.x₀
    ssm_mats_fn = dt -> ImplicitEulerJointSSMMatrices(ssm, dt)
    return joint_ssm(x₀, ssm_mats_fn, ssm.ts)
end
