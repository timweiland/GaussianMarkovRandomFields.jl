using LinearAlgebra, LinearMaps
import LinearAlgebra: issymmetric

import Base: size

export OuterProductMap, to_matrix

"""
    OuterProductMap{T}(A, Q)

Represents the outer product A' Q A, without actually forming it in memory.
"""
mutable struct OuterProductMap{T} <: LinearMaps.LinearMap{T}
    A::LinearMap{T}
    Q::LinearMap{T}
    to_mat_cache::Union{Nothing,AbstractMatrix{T}}

    function OuterProductMap(A::LinearMap{T}, Q::LinearMap{T}) where {T}
        Base.size(Q, 1) == Base.size(Q, 2) || throw(ArgumentError("Q must be square"))
        Base.size(A, 1) == Base.size(Q, 1) || throw(ArgumentError("size mismatch"))
        new{T}(A, Q, nothing)
    end
end

size(d::OuterProductMap) = Base.size(d.A, 2), Base.size(d.A, 2)

function LinearMaps._unsafe_mul!(y, L::OuterProductMap, x::AbstractVector)
    y .= L.A' * (L.Q * (L.A * x))
end

LinearAlgebra.adjoint(L::OuterProductMap) = OuterProductMap(L.A, L.Q')
LinearAlgebra.transpose(L::OuterProductMap) = OuterProductMap(L.A, L.Q')
LinearAlgebra.issymmetric(L::OuterProductMap) = LinearAlgebra.issymmetric(L.Q)

function to_matrix(L::OuterProductMap)
    if L.to_mat_cache !== nothing
        return L.to_mat_cache
    end
    A_mat = to_matrix(L.A)

    if L.Q isa LinearMaps.UniformScalingMap
        L.to_mat_cache = Symmetric(L.Q.λ * A_mat' * A_mat)
        return L.to_mat_cache
    end
    Q_mat = to_matrix(L.Q)
    L.to_mat_cache = A_mat' * Q_mat * A_mat
    if issymmetric(Q_mat)
        L.to_mat_cache = Symmetric(L.to_mat_cache)
    end
    return L.to_mat_cache
end

function linmap_sqrt(OP::OuterProductMap)
    if !issymmetric(OP)
        throw(ArgumentError("Map is not symmetric"))
    end

    if OP.Q isa LinearMaps.UniformScalingMap
        return sqrt(OP.Q.λ) * OP.A'
    end
    return OP.A' * linmap_sqrt(OP.Q)
end
