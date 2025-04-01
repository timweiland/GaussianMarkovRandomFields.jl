using LinearAlgebra
using LinearMaps
using SparseArrays

export CholeskyFactorizedMap

"""
    CholeskyFactorizedMap{T}(cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}})

A linear map represented in terms of its Cholesky factorization,
i.e. for `A = L * L'`, this map represents `A`.

# Arguments
- `cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}}`:
    The Cholesky factorization of a symmetric positive definite matrix.
"""
struct CholeskyFactorizedMap{T} <: LinearMap{T}
    cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}}
    sqrt_map::CholeskySqrt{T}

    function CholeskyFactorizedMap(
        cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}}
    ) where {T}
        sqrt_map = CholeskySqrt(cho)
        new{T}(cho, sqrt_map)
    end
end

size(C::CholeskyFactorizedMap) = Base.size(C.cho)

function LinearMaps._unsafe_mul!(y, C::CholeskyFactorizedMap, x::AbstractVector)
    mul!(y, C.sqrt_map, C.sqrt_map' * x)
end

LinearAlgebra.adjoint(C::CholeskyFactorizedMap) = C
LinearAlgebra.transpose(C::CholeskyFactorizedMap) = C
LinearAlgebra.issymmetric(C::CholeskyFactorizedMap) = true
LinearAlgebra.isposdef(C::CholeskyFactorizedMap) = true

function to_matrix(C::CholeskyFactorizedMap)
    if C.cho isa Cholesky
        return Array(C.cho)
    end
    return sparse(C.cho)
end

linmap_sqrt(C::CholeskyFactorizedMap) = C.sqrt_map

function linmap_cholesky(C::CholeskyFactorizedMap; perm=nothing)
    if perm !== nothing
        @warn "User-specified permutation for Cholesky of CholeskyFactorizedMap!"
    end
    return C.cho
end

function Base.show(io::IO, C::CholeskyFactorizedMap)
    println(
        io,
        "$(Base.size(C, 1))x$(Base.size(C, 2)) CholeskyFactorizedMap",
    )
    print(io, C.cho)
end
