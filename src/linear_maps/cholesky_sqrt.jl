using LinearAlgebra, SparseArrays
import LinearMaps: _unsafe_mul!

export CholeskySqrt, DenseCholeskySqrt, SparseCholeskySqrt

function sparse_cho_sqrt(cho::SparseArrays.CHOLMOD.Factor{Tv, Ti}) where {Tv, Ti}
	s = unsafe_load(pointer(cho))
    p⁻¹ = invperm(cho.p)
    if Bool(s.is_ll)
        return sparse(cho.L)[p⁻¹, :]
    end
	LD = sparse(cho.LD)
    L, d = SparseArrays.CHOLMOD.getLd!(LD)
    return (L .* d')[p⁻¹, :]
end

"""
    CholeskySqrt(cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}})

A linear map representing the square root obtained from a Cholesky
factorization, i.e. for `A = L * L'`, this map represents `L`.

# Arguments
- `cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}}`:
    The Cholesky factorization of a symmetric positive definite matrix.
"""
CholeskySqrt(cho::Cholesky{T}) where {T} = DenseCholeskySqrt{T}(cho)
CholeskySqrt(cho::SparseArrays.CHOLMOD.Factor) = SparseCholeskySqrt(cho)

struct DenseCholeskySqrt{T} <: LinearMap{T}
    cho::Cholesky{T}
end

struct SparseCholeskySqrt{Tv, Ti} <: LinearMap{Tv}
    cho::SparseArrays.CHOLMOD.Factor{Tv, Ti}
    L_sparse::SparseMatrixCSC{Tv}
end

function SparseCholeskySqrt(cho::SparseArrays.CHOLMOD.Factor{Tv, Ti}) where {Tv, Ti}
    L_sparse = sparse_cho_sqrt(cho)
    return SparseCholeskySqrt{Tv, Ti}(cho, L_sparse)
end

function LinearMaps._unsafe_mul!(y, L::DenseCholeskySqrt, x::AbstractVector)
    mul!(y, L.cho.L, x)
end

function LinearMaps._unsafe_mul!(y, L::SparseCholeskySqrt, x::AbstractVector)
    mul!(y, L.L_sparse, x)
end

function LinearMaps._unsafe_mul!(
    y,
    transL::LinearMaps.TransposeMap{<:Any,<:DenseCholeskySqrt},
    x::AbstractVector,
)
    mul!(y, transL.lmap.cho.U, x)
end

function LinearMaps._unsafe_mul!(
    y,
    transL::LinearMaps.TransposeMap{<:Any,<:SparseCholeskySqrt},
    x::AbstractVector,
)
    mul!(y, transL.lmap.L_sparse', x)
end

LinearAlgebra.size(L::Union{DenseCholeskySqrt, SparseCholeskySqrt}) = size(L.cho)

to_matrix(L::DenseCholeskySqrt) = L.cho.L
to_matrix(L::SparseCholeskySqrt) = L.L_sparse


linmap_sqrt(A::LinearMaps.WrappedMap) = CholeskySqrt(cholesky(to_matrix(A)))
