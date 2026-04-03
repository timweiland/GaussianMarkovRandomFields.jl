module GaussianMarkovRandomFieldsAutoDiff

using GaussianMarkovRandomFields
using ForwardDiff, Zygote, LinearAlgebra, LinearMaps, SparseArrays
import LinearMaps: _unsafe_mul!

#                                  |    |    |
#                                 )_)  )_)  )_)
#                                )___))___))___)
#                               )____)____)_____)
#                             _____|____|____|____
#                    ---------\                   /---------
#                      ^^^^^ ^^^^^^^^^^^^^^^^^^^^^
#                        ^^^^      ^^^^     ^^^    ^^
#                               ^^^^      ^^^
#
# Zygote accum for sparse Hermitian/Symmetric (piracy until upstream PR is merged)
const HermOrSymSparse{T, I} = Union{
    Hermitian{T, SparseMatrixCSC{T, I}},
    Symmetric{T, SparseMatrixCSC{T, I}},
}
Zygote.accum(x::HermOrSymSparse, y::HermOrSymSparse) = x + y

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
