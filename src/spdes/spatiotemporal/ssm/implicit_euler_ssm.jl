using SparseArrays

export ImplicitEulerSSM, ImplicitEulerJointSSMMatrices

"""
    ImplicitEulerSSM(
        x₀::AbstractGMRF,
        G::Function,
        M::Function,
        M⁻¹::Function,
        β::Function,
        β⁻¹::Function,
        spatial_noise::AbstractGMRF,
        ts::AbstractVector,
        constraint_handler::ConstraintHandler,
        constraint_noise::AbstractVector,
    )

State-space model for the implicit Euler discretization of a stochastic
differential equation.

The state-space model is given by

```
G(Δt) xₖ₊₁ = M(Δt) xₖ + M(Δt) β(Δt) zₛ
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
    constraint_handler::ConstraintHandler
    constraint_noise::AbstractVector

    function ImplicitEulerSSM(
        x₀,
        G,
        M,
        M⁻¹,
        β,
        β⁻¹,
        spatial_noise,
        ts::AbstractVector,
        constraint_handler,
        constraint_noise,
    )
        new(
            x₀,
            G,
            M,
            M⁻¹,
            β,
            β⁻¹,
            spatial_noise,
            ts::AbstractVector,
            constraint_handler,
            constraint_noise,
        )
    end
end

"""
    ImplicitEulerJointSSMMatrices(
        ssm::ImplicitEulerSSM,
        Δt::Real
    )

Construct the joint state-space model matrices for the implicit Euler
discretization scheme.

# Arguments
- `ssm::ImplicitEulerSSM`: The implicit Euler state-space model.
- `Δt::Real`: The time step.
"""
struct ImplicitEulerJointSSMMatrices <: JointSSMMatrices
    Δt::Real
    G::AbstractMatrix
    M::AbstractMatrix
    Σ⁻¹::AbstractMatrix
    Σ⁻¹_sqrt::AbstractMatrix
    constraint_handler::ConstraintHandler
    constraint_noise::AbstractVector

    function ImplicitEulerJointSSMMatrices(ssm::ImplicitEulerSSM, Δt::Real)
        G = sparse(ssm.G(Δt))
        M⁻¹ = sparse(ssm.M⁻¹(Δt))
        M = sparse(ssm.M(Δt))
        β⁻¹ = ssm.β⁻¹(Δt)
        Q_s = precision_map(ssm.spatial_noise)
        Q_s_sqrt = to_matrix(linmap_sqrt(Q_s))
        Q_s = to_matrix(Q_s)

        Σ⁻¹ = M⁻¹' * β⁻¹' * Q_s * β⁻¹ * M⁻¹
        Σ⁻¹_sqrt = M⁻¹' * β⁻¹' * Q_s_sqrt

        return new(Δt, G, M, Σ⁻¹, Σ⁻¹_sqrt, ssm.constraint_handler, ssm.constraint_noise)
    end
end

function joint_ssm(ssm::ImplicitEulerSSM)
    x₀ = ssm.x₀
    ssm_mats_fn = dt -> ImplicitEulerJointSSMMatrices(ssm, dt)
    return joint_ssm(x₀, ssm_mats_fn, ssm.ts)
end
