using LinearAlgebra
using LinearMaps
using SparseArrays

import Base: getproperty

export CholeskyFactorizedMap

"""
    CholeskyFactorizedMap{T,C}(cho::C) where {T,C}

A linear map represented in terms of its Cholesky factorization,
i.e. for `A = L * L'`, this map represents `A`.

# Type Parameters
- `T`: Element type of the matrix
- `C`: Type of the Cholesky factorization

# Arguments
- `cho`: The Cholesky factorization of a symmetric positive definite matrix.
  Can be `Cholesky`, `SparseArrays.CHOLMOD.Factor`, or `LDLFactorization`.
"""
mutable struct CholeskyFactorizedMap{T,C} <: LinearMap{T}
    cho::C
    _sqrt_map_cache::Union{Nothing, SparseCholeskySqrt{T}, DenseCholeskySqrt{T}}

    function CholeskyFactorizedMap(cho::C) where {T,C<:Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}}}
        new{T,C}(cho, nothing)
    end
    
    # Constructor for LDLFactorization (from LDLFactorizations.jl extension)
    function CholeskyFactorizedMap{T}(cho::C) where {T,C}
        new{T,C}(cho, nothing)
    end
end

function getproperty(C::CholeskyFactorizedMap, sym::Symbol)
    if sym === :sqrt_map
        if C._sqrt_map_cache === nothing
            C._sqrt_map_cache = CholeskySqrt(C.cho)
        end
        return C._sqrt_map_cache
    else
        return getfield(C, sym)
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

function linmap_cholesky(::Val{:default}, C::CholeskyFactorizedMap{T,<:Union{Cholesky,SparseArrays.CHOLMOD.Factor}}; perm=nothing) where {T}
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
