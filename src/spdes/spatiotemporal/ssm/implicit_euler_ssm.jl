using SparseArrays
import Ferrite: ConstraintHandler

export ImplicitEulerSSM, ImplicitEulerJointSSMMatrices

"""
    ImplicitEulerSSM{X,S,GF,MF,MIF,BF,BIF,TS,C,V}(
        x₀::X,
        G::GF,
        M::MF,
        M⁻¹::MIF,
        β::BF,
        β⁻¹::BIF,
        spatial_noise::S,
        ts::TS,
        constraint_handler::C,
        constraint_noise::V,
    )

State-space model for the implicit Euler discretization of a stochastic
differential equation.

The state-space model is given by

```
G(Δt) xₖ₊₁ = M(Δt) xₖ + M(Δt) β(Δt) zₛ
```

where `zₛ` is (possibly colored) spatial noise. 
"""
struct ImplicitEulerSSM{X <: AbstractGMRF, S <: AbstractGMRF, GF <: Function, MF <: Function, MIF <: Function, BF <: Function, BIF <: Function, TS <: AbstractVector{<:Real}, C <: ConstraintHandler, V <: AbstractVector}
    x₀::X
    G::GF
    M::MF
    M⁻¹::MIF
    β::BF
    β⁻¹::BIF
    spatial_noise::S
    ts::TS
    constraint_handler::C
    constraint_noise::V

    function ImplicitEulerSSM(
            x₀::X,
            G::GF,
            M::MF,
            M⁻¹::MIF,
            β::BF,
            β⁻¹::BIF,
            spatial_noise::S,
            ts::TS,
            constraint_handler::C,
            constraint_noise::V,
        ) where {X <: AbstractGMRF, S <: AbstractGMRF, GF <: Function, MF <: Function, MIF <: Function, BF <: Function, BIF <: Function, TS <: AbstractVector{<:Real}, C <: ConstraintHandler, V <: AbstractVector}
        return new{X, S, GF, MF, MIF, BF, BIF, TS, C, V}(
            x₀,
            G,
            M,
            M⁻¹,
            β,
            β⁻¹,
            spatial_noise,
            ts,
            constraint_handler,
            constraint_noise,
        )
    end
end

"""
    ImplicitEulerJointSSMMatrices{T,GM,MM,SM,SQRT,C,V}(
        ssm::ImplicitEulerSSM,
        Δt::Real
    )

Construct the joint state-space model matrices for the implicit Euler
discretization scheme.

# Arguments
- `ssm::ImplicitEulerSSM`: The implicit Euler state-space model.
- `Δt::Real`: The time step.
"""
struct ImplicitEulerJointSSMMatrices{T <: Real, GM <: AbstractMatrix, MM <: AbstractMatrix, SM <: AbstractMatrix, SQRT <: Union{Nothing, AbstractMatrix}, C <: ConstraintHandler, V <: AbstractVector} <: JointSSMMatrices
    Δt::T
    G::GM
    M::MM
    Σ⁻¹::SM
    Σ⁻¹_sqrt::SQRT
    constraint_handler::C
    constraint_noise::V

    function ImplicitEulerJointSSMMatrices(ssm::ImplicitEulerSSM, Δt::Real)
        G = sparse(ssm.G(Δt))
        M⁻¹ = sparse(ssm.M⁻¹(Δt))
        M = sparse(ssm.M(Δt))
        β⁻¹ = ssm.β⁻¹(Δt)
        Q_s = precision_map(ssm.spatial_noise)
        Q_s = to_matrix(Q_s)

        # Use Q_sqrt from spatial_noise if available, otherwise pass nothing
        Q_s_sqrt = if ssm.spatial_noise.Q_sqrt !== nothing
            to_matrix(ssm.spatial_noise.Q_sqrt)
        else
            nothing
        end

        Σ⁻¹ = M⁻¹' * β⁻¹' * Q_s * β⁻¹ * M⁻¹
        Σ⁻¹_sqrt = if Q_s_sqrt !== nothing
            M⁻¹' * β⁻¹' * Q_s_sqrt
        else
            nothing
        end

        return new{typeof(Δt), typeof(G), typeof(M), typeof(Σ⁻¹), typeof(Σ⁻¹_sqrt), typeof(ssm.constraint_handler), typeof(ssm.constraint_noise)}(
            Δt, G, M, Σ⁻¹, Σ⁻¹_sqrt, ssm.constraint_handler, ssm.constraint_noise
        )
    end
end

function joint_ssm(ssm::ImplicitEulerSSM)
    x₀ = ssm.x₀
    ssm_mats_fn = dt -> ImplicitEulerJointSSMMatrices(ssm, dt)
    return joint_ssm(x₀, ssm_mats_fn, ssm.ts)
end
