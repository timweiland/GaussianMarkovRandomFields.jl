using LinearAlgebra, LinearMaps

export ADJacobianMap, ADJacobianAdjointMap

"""
    ADJacobianMap(f::Function, x₀::AbstractVector{T}, N_outputs::Int)

A linear map representing the Jacobian of `f` at `x₀`.
Uses forward-mode AD in a matrix-free way, i.e. we do not actually store
the Jacobian in memory and only compute JVPs.

Requires ForwardDiff.jl!

# Arguments
- `f::Function`: Function to differentiate.
- `x₀::AbstractVector{T}`: Input vector at which to evaluate the Jacobian.
- `N_outputs::Int`: Output dimension of `f`.
"""
struct ADJacobianMap{T} <: LinearMaps.LinearMap{T}
    f::Function
    x₀::AbstractVector{T}
    N_outputs::Int

    function ADJacobianMap(f::Function, x₀::AbstractVector{T}, N_outputs::Int) where {T}
        N_outputs > 0 || throw(ArgumentError("N_outputs must be positive"))
        new{T}(f, x₀, N_outputs)
    end
end

function LinearMaps.size(J::ADJacobianMap)
    return (J.N_outputs, length(J.x₀))
end

LinearAlgebra.adjoint(J::ADJacobianMap) = ADJacobianAdjointMap(J.f, J.x₀, J.N_outputs)
LinearAlgebra.transpose(J::ADJacobianMap) = ADJacobianAdjointMap(J.f, J.x₀, J.N_outputs)

"""
    ADJacobianAdjointMap{T}(f::Function, x₀::AbstractVector{T}, N_outputs::Int)

A linear map representing the adjoint of the Jacobian of `f` at `x₀`.
Uses reverse-mode AD in a matrix-free way, i.e. we do not actually store
the Jacobian in memory and only compute VJPs.

Requires Zygote.jl!

# Arguments
- `f::Function`: Function to differentiate.
- `x₀::AbstractVector{T}`: Input vector at which to evaluate the Jacobian.
- `N_outputs::Int`: Output dimension of `f`.
"""
struct ADJacobianAdjointMap{T} <: LinearMaps.LinearMap{T}
    f::Function
    x₀::AbstractVector{T}
    N_outputs::Int
    f_val::Union{Real,AbstractVector}
    f_pullback::Function
end

function LinearMaps.size(J::ADJacobianAdjointMap)
    return (length(J.x₀), J.N_outputs)
end

LinearAlgebra.adjoint(J::ADJacobianAdjointMap) = ADJacobianMap(J.f, J.x₀, J.N_outputs)
LinearAlgebra.transpose(J::ADJacobianAdjointMap) = ADJacobianMap(J.f, J.x₀, J.N_outputs)
