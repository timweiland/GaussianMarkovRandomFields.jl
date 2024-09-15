using LinearAlgebra, LinearMaps, Symbolics, SparseDiffTools
import LinearMaps: _unsafe_mul!

export SparseJacobianMap, SparseJacobianAdjointMap

struct SparseJacobianMap{T} <: LinearMaps.LinearMap{T}
    f!::Function
    x₀::AbstractVector{T}
    N_outputs::Int
    jac::AbstractMatrix{T}

    function SparseJacobianMap(
        f!::Function,
        x₀::AbstractVector{T},
        N_outputs::Int,
        colors = nothing,
        jac = nothing,
    ) where {T}
        N_outputs > 0 || throw(ArgumentError("N_outputs must be positive"))

        output = rand(N_outputs)

        if colors === nothing
            sparsity_pattern = Symbolics.jacobian_sparsity(f!, output, x₀)
            jac = Float64.(sparsity_pattern)
            colors = matrix_colors(jac)
        end
        forwarddiff_color_jacobian!(jac, f!, x₀, colorvec = colors)

        new{T}(f!, x₀, N_outputs, jac)
    end
end

function LinearMaps._unsafe_mul!(y, J::SparseJacobianMap, x::AbstractVector)
    y .= J.jac * x
end

function LinearMaps.size(J::SparseJacobianMap)
    return (J.N_outputs, length(J.x₀))
end

LinearAlgebra.adjoint(J::SparseJacobianMap) =
    SparseJacobianAdjointMap(J.f!, J.x₀, J.N_outputs, J.jac)
LinearAlgebra.transpose(J::SparseJacobianMap) =
    SparseJacobianAdjointMap(J.f!, J.x₀, J.N_outputs, J.jac)

struct SparseJacobianAdjointMap{T} <: LinearMaps.LinearMap{T}
    f!::Function
    x₀::AbstractVector{T}
    N_outputs::Int
    jac::AbstractMatrix{T}

    function SparseJacobianAdjointMap(
        f!::Function,
        x₀::AbstractVector{T},
        N_outputs::Int,
        jac::AbstractMatrix{T},
    ) where {T}
        N_outputs > 0 || throw(ArgumentError("N_outputs must be positive"))
        new{T}(f!, x₀, N_outputs, jac)
    end
end

function LinearMaps._unsafe_mul!(y, J::SparseJacobianAdjointMap, x::AbstractVector)
    y .= J.jac' * x
end

function LinearMaps.size(J::SparseJacobianAdjointMap)
    return (length(J.x₀), J.N_outputs)
end
