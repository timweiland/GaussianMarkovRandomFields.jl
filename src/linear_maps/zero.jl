using LinearMaps, SparseArrays

export ZeroMap

"""
    ZeroMap{T}(N::Int, M::Int)

A linear map that maps all vectors to the zero vector.

# Arguments
- `N::Int`: Output dimension
- `M::Int`: Input dimension
"""
struct ZeroMap{T} <: LinearMaps.LinearMap{T}
    N::Int
    M::Int
end

function LinearMaps._unsafe_mul!(y, L::ZeroMap, x::AbstractVector)
    y .= 0
end

function LinearMaps.size(L::ZeroMap)
    return (L.N, L.M)
end

function LinearMaps.adjoint(L::ZeroMap{T}) where {T}
    return ZeroMap{T}(L.M, L.N)
end

function LinearMaps.transpose(L::ZeroMap{T}) where {T}
    return ZeroMap{T}(L.M, L.N)
end

function to_matrix(L::ZeroMap)
    return spzeros(L.N, L.M)
end

function linmap_sqrt(L::ZeroMap{T}) where {T}
    return ZeroMap{T}(L.N, L.M)
end
