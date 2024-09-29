using LinearAlgebra, LinearMaps
import SparseArrays: sparse

import Base: size

export OuterProductMap, to_matrix

struct OuterProductMap{T} <: LinearMaps.LinearMap{T}
    A::LinearMap{T}
    Q::LinearMap{T}

    function OuterProductMap(A::LinearMap{T}, Q::LinearMap{T}) where {T}
        Base.size(Q, 1) == Base.size(Q, 2) || throw(ArgumentError("Q must be square"))
        Base.size(A, 1) == Base.size(Q, 1) || throw(ArgumentError("size mismatch"))
        new{T}(A, Q)
    end
end

size(d::OuterProductMap) = Base.size(d.A, 2), Base.size(d.A, 2)

function LinearMaps._unsafe_mul!(y, L::OuterProductMap, x::AbstractVector)
    y .= L.A' * (L.Q * (L.A * x))
end

LinearAlgebra.adjoint(L::OuterProductMap) = L
LinearAlgebra.transpose(L::OuterProductMap) = L

function to_matrix(L::OuterProductMap)
    A_mat = to_matrix(L.A)

    if L.Q isa LinearMaps.UniformScalingMap
        return Symmetric(L.Q.λ * A_mat' * A_mat)
    end
    Q_mat = to_matrix(L.Q)
    return Symmetric(A_mat' * Q_mat * A_mat)
end
