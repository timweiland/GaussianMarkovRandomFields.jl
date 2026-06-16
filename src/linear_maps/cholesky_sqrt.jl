using LinearAlgebra, SparseArrays
import LinearMaps: _unsafe_mul!

export CholeskySqrt

function sparse_cho_sqrt(cho::SparseArrays.CHOLMOD.Factor)
    s = unsafe_load(pointer(cho))
    p⁻¹ = invperm(cho.p)
    if Bool(s.is_ll)
        return sparse(cho.L)[p⁻¹, :]
    end
    # `sparse(::FactorComponent{_,:LD})` is inferred as a non-concrete union
    # (`Union{SparseMatrixCSC, Symmetric}` — CHOLMOD's `sparse` can't be type-stable across
    # `stype`), but an :LD component always has `stype == 0`, so it is concretely a
    # `SparseMatrixCSC`. Assert that so `getLd!` (which has only a `::SparseMatrixCSC`
    # method) resolves without a union-split (a no-method JET flags on Julia ≥1.11).
    LD = sparse(cho.LD)::SparseMatrixCSC
    L, d = SparseArrays.CHOLMOD.getLd!(LD)
    return (L .* sqrt.(d)')[p⁻¹, :]
end

"""
    CholeskySqrt(cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}})

A linear map representing the square root obtained from a Cholesky
factorization, i.e. for `A = L * L'`, this map represents `L`.

# Arguments
- `cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}}`:
    The Cholesky factorization of a symmetric positive definite matrix.
"""
CholeskySqrt(cho::Cholesky) = LinearMap(cho.L)
CholeskySqrt(cho::SparseArrays.CHOLMOD.Factor) = LinearMap(sparse_cho_sqrt(cho))


linmap_sqrt(A::LinearMaps.WrappedMap) = CholeskySqrt(cholesky(to_matrix(A)))
