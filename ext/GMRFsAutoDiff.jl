module GMRFsAutoDiff

using GMRFs
using ForwardDiff, Zygote, LinearAlgebra, LinearMaps
import LinearMaps: _unsafe_mul!

function LinearMaps._unsafe_mul!(y, J::ADJacobianMap, x::AbstractVector)
    g(t) = J.f(J.x₀ + t * x)
    y .= ForwardDiff.derivative(g, 0.0)
end

function GMRFs.ADJacobianAdjointMap(
    f::Function,
    x₀::AbstractVector{T},
    N_outputs::Int,
) where {T}
    N_outputs > 0 || throw(ArgumentError("N_outputs must be positive"))
    f_val, f_pullback = Zygote.pullback(f, x₀)
    ADJacobianAdjointMap{T}(f, x₀, N_outputs, f_val, f_pullback)
end

function LinearMaps._unsafe_mul!(y, J::ADJacobianAdjointMap, x::AbstractVector)
    y .= J.f_pullback(x)[1]
end

end
