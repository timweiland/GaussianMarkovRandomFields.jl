export LinearMapWithSqrt
import LinearMaps: _unsafe_mul!
import LinearAlgebra: issymmetric
import Base: kron

"""
    LinearMapWithSqrt{T}(
        A::LinearMap{T},
        A_sqrt::LinearMap{T},
    )

A symmetric positive definite linear map `A` with known square root `A_sqrt`,
i.e. `A = A_sqrt * A_sqrt'`.
Behaves just like `A`, but taking the square root directly returns `A_sqrt`.

# Arguments
- `A::LinearMap{T}`: The linear map `A`.
- `A_sqrt::LinearMap{T}`: The square root of `A`.
"""
mutable struct LinearMapWithSqrt{T, L1, L2, M<:AbstractMatrix{T}} <: LinearMaps.LinearMap{T}
    A::L1
    A_sqrt::L2
    A_mat_cache::Union{Nothing,M}

    function LinearMapWithSqrt(
        A::L1,
        A_sqrt::L2,
    ) where {T, L1<:LinearMaps.LinearMap{T}, L2<:LinearMaps.LinearMap{T}}
        Base.size(A, 1) == Base.size(A_sqrt, 1) || throw(ArgumentError("size mismatch"))
        new{T, L1, L2, SparseMatrixCSC{T, Int}}(A, A_sqrt, nothing)
    end
end

function LinearMaps._unsafe_mul!(y, L::LinearMapWithSqrt, x::AbstractVector)
    LinearMaps._unsafe_mul!(y, L.A, x)
end

function LinearMaps.size(L::LinearMapWithSqrt)
    return LinearMaps.size(L.A)
end

function LinearMaps.adjoint(L::LinearMapWithSqrt)
    return L
end

function LinearMaps.transpose(L::LinearMapWithSqrt)
    return L
end

LinearAlgebra.issymmetric(::LinearMapWithSqrt) = true

function to_matrix(L::LinearMapWithSqrt{<:Any, <:Any, <:Any, M})::M where {M}
    if L.A_mat_cache === nothing
        L.A_mat_cache = to_matrix(L.A)
    end
    return L.A_mat_cache
end

function linmap_sqrt(L::LinearMapWithSqrt)
    return L.A_sqrt
end

function Base.kron(A::LinearMapWithSqrt, B::LinearMapWithSqrt)
    return LinearMapWithSqrt(Base.kron(A.A, B.A), Base.kron(A.A_sqrt, B.A_sqrt))
end
