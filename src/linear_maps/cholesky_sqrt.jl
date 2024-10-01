using LinearAlgebra, SparseArrays
import LinearMaps: _unsafe_mul!

export CholeskySqrt

struct CholeskySqrt{T} <: LinearMap{T}
    cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}}
    sqrt_map::Function
    sqrt_adjoint_map::Function

    function CholeskySqrt(cho::Cholesky{T}) where {T}
        sqrt_map = (x) -> cho.L * x
        sqrt_adjoint_map = (x) -> cho.U * x
        new{T}(cho, sqrt_map, sqrt_adjoint_map)
    end

    function CholeskySqrt(cho::SparseArrays.CHOLMOD.Factor{T}) where {T}
        p = cho.p
        p⁻¹ = invperm(p)
        L_sp = sparse(cho.L)
        sqrt_map = (x) -> (L_sp*x)[p⁻¹]
        sqrt_adjoint_map = (x) -> (L_sp' * x[p])
        new{T}(cho, sqrt_map, sqrt_adjoint_map)
    end
end

function LinearMaps._unsafe_mul!(y, L::CholeskySqrt, x::AbstractVector)
    y .= L.sqrt_map(x)
end

function LinearMaps._unsafe_mul!(
    y,
    transL::LinearMaps.TransposeMap{<:Any,<:CholeskySqrt},
    x::AbstractVector,
)
    y .= transL.lmap.sqrt_adjoint_map(x)
end

function LinearAlgebra.size(L::CholeskySqrt)
    return size(L.cho, 1), size(L.cho, 1)
end

function to_matrix(L::CholeskySqrt)
    if L.cho isa Cholesky
        return L.cho.L
    else
        return sparse(L.cho.L)[invperm(L.cho.p), :]
    end
end
