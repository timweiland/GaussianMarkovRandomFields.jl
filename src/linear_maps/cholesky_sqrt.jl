using LinearAlgebra, SparseArrays
import LinearMaps: _unsafe_mul!

export CholeskySqrt

function sparse_cho_sqrt(cho::SparseArrays.CHOLMOD.Factor)
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
    CholeskySqrt{T}(cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}})

A linear map representing the square root obtained from a Cholesky
factorization, i.e. for `A = L * L'`, this map represents `L`.

# Arguments
- `cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}}`:
    The Cholesky factorization of a symmetric positive definite matrix.
"""
struct CholeskySqrt{T} <: LinearMap{T}
    cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}}
    sqrt_map::Function
    sqrt_adjoint_map::Function

    function CholeskySqrt(cho::Cholesky{T}) where {T}
        sqrt_map = (y, x) -> mul!(y, cho.L, x)
        sqrt_adjoint_map = (y, x) -> mul!(y, cho.U, x)
        new{T}(cho, sqrt_map, sqrt_adjoint_map)
    end

    function CholeskySqrt(cho::SparseArrays.CHOLMOD.Factor{T}) where {T}
        L_sp = sparse_cho_sqrt(cho)
        sqrt_map = (y, x) -> mul!(y, L_sp, x)
        sqrt_adjoint_map = (y, x) -> mul!(y, L_sp', x)
        new{T}(cho, sqrt_map, sqrt_adjoint_map)
    end
end

function LinearMaps._unsafe_mul!(y, L::CholeskySqrt, x::AbstractVector)
    L.sqrt_map(y, x)
end

function LinearMaps._unsafe_mul!(
    y,
    transL::LinearMaps.TransposeMap{<:Any,<:CholeskySqrt},
    x::AbstractVector,
)
    transL.lmap.sqrt_adjoint_map(y, x)
end

function LinearAlgebra.size(L::CholeskySqrt)
    return size(L.cho)
end

function to_matrix(L::CholeskySqrt)
    if L.cho isa Cholesky
        return L.cho.L
    else
        return sparse(L.cho.L)[invperm(L.cho.p), :]
    end
end

linmap_sqrt(A::LinearMaps.WrappedMap) = CholeskySqrt(cholesky(to_matrix(A)))
