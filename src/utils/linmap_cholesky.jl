using LinearMaps, LinearAlgebra

export linmap_cholesky

"""
    linmap_cholesky(A::LinearMap; perm=nothing)

Compute the Cholesky factorization `A = L * L'` (with lower-triangular L),
optionally up to a permutation `perm`.

# Returns
A Cholesky factorization of type `Union{Cholesky, SparseArrays.CHOLMOD.Factor}`.
"""
function linmap_cholesky(A::LinearMap; perm=nothing)
    if issymmetric(A)
        if perm !== nothing
            return cholesky(to_matrix(A); perm=perm)
        end
        return cholesky(to_matrix(A))
    end
    throw(ArgumentError("$A is not symmetric; Cholesky not available"))
end
