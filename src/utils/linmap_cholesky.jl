using LinearMaps, LinearAlgebra

export linmap_cholesky

function _ensure_symmetry(A::LinearMap)
    if !issymmetric(A)
        throw(ArgumentError("$A is not symmetric; Cholesky not available"))
    end
end

"""
    linmap_cholesky(A::LinearMap; perm=nothing)

Compute the Cholesky factorization `A = L * L'` (with lower-triangular L),
optionally up to a permutation `perm`.

# Returns
A Cholesky factorization.
"""
function linmap_cholesky(::Val{:default}, A::LinearMap; perm=nothing)
    _ensure_symmetry(A)
    return linmap_cholesky_default(to_matrix(A); perm=perm)
end

function linmap_cholesky(::Val{:autodiffable}, A::LinearMap; perm=nothing)
    _ensure_symmetry(A)
    return linmap_cholesky_ldl_factorizations(sparse(to_matrix(A)); perm=perm)
end

function linmap_cholesky_default(A::AbstractMatrix; perm=nothing)
    if perm !== nothing
        return cholesky(to_matrix(A); perm=perm)
    end
    return cholesky(A)
end

function linmap_cholesky_ldl_factorizations(::AbstractMatrix; perm=nothing)
    error("LDLFactorizations.jl must be loaded to use it for factorization.")
end
