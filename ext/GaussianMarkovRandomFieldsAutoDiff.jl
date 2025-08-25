module GaussianMarkovRandomFieldsAutoDiff

using GaussianMarkovRandomFields
using ForwardDiff, Zygote, LinearAlgebra, LinearMaps
import LinearMaps: _unsafe_mul!

function LinearMaps._unsafe_mul!(y, J::ADJacobianMap, x::AbstractVector)
    g(t) = J.f(J.x₀ + t * x)
    return y .= ForwardDiff.derivative(g, 0.0)
end

function GaussianMarkovRandomFields.ADJacobianAdjointMap(
        f::Function,
        x₀::AbstractVector{T},
        N_outputs::Int,
    ) where {T}
    N_outputs > 0 || throw(ArgumentError("N_outputs must be positive"))
    f_val, f_pullback = Zygote.pullback(f, x₀)
    return ADJacobianAdjointMap{T}(f, x₀, N_outputs, f_val, f_pullback)
end

function LinearMaps._unsafe_mul!(y, J::ADJacobianAdjointMap, x::AbstractVector)
    return y .= J.f_pullback(x)[1]
end

end
