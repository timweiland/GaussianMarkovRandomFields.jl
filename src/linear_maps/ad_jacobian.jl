using ForwardDiff, Zygote, LinearAlgebra, LinearMaps
import LinearMaps: _unsafe_mul!

export ADJacobianMap, ADJacobianAdjointMap

"""
    ADJacobianMap(f, x₀, N_outputs)

A linear map representing the Jacobian of `f` at `x₀`.
Uses forward-mode AD in a matrix-free way, i.e. we do not actually store
the Jacobian in memory and only compute JVPs.
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

function LinearMaps._unsafe_mul!(y, J::ADJacobianMap, x::AbstractVector)
    g(t) = J.f(J.x₀ + t * x)
    y .= ForwardDiff.derivative(g, 0.0)
end

function LinearMaps.size(J::ADJacobianMap)
    return (J.N_outputs, length(J.x₀))
end

LinearAlgebra.adjoint(J::ADJacobianMap) = ADJacobianAdjointMap(J.f, J.x₀, J.N_outputs)
LinearAlgebra.transpose(J::ADJacobianMap) = ADJacobianAdjointMap(J.f, J.x₀, J.N_outputs)

"""
    ADJacobianAdjointMap(f, x₀, N_outputs)

A linear map representing the adjoint of the Jacobian of `f` at `x₀`.
Uses reverse-mode AD in a matrix-free way, i.e. we do not actually store
the Jacobian in memory and only compute VJPs.
"""
struct ADJacobianAdjointMap{T} <: LinearMaps.LinearMap{T}
    f::Function
    x₀::AbstractVector{T}
    N_outputs::Int
    f_val::Union{Real,AbstractVector}
    f_pullback::Function

    function ADJacobianAdjointMap(
        f::Function,
        x₀::AbstractVector{T},
        N_outputs::Int,
    ) where {T}
        N_outputs > 0 || throw(ArgumentError("N_outputs must be positive"))
        f_val, f_pullback = Zygote.pullback(f, x₀)
        new{T}(f, x₀, N_outputs, f_val, f_pullback)
    end
end

function LinearMaps._unsafe_mul!(y, J::ADJacobianAdjointMap, x::AbstractVector)
    y .= J.f_pullback(x)[1]
end

function LinearMaps.size(J::ADJacobianAdjointMap)
    return (length(J.x₀), J.N_outputs)
end

LinearAlgebra.adjoint(J::ADJacobianAdjointMap) = ADJacobianMap(J.f, J.x₀, J.N_outputs)
LinearAlgebra.transpose(J::ADJacobianAdjointMap) = ADJacobianMap(J.f, J.x₀, J.N_outputs)
