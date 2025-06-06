using LinearMaps, LinearAlgebra

export linmap_cholesky

"""
    linmap_cholesky(A::LinearMap; perm=nothing)

Compute the Cholesky factorization `A = L * L'` (with lower-triangular L),
optionally up to a permutation `perm`.

# Returns
A Cholesky factorization.
"""
function linmap_cholesky(A::LinearMap; perm=nothing, method=:default)
    if issymmetric(A)
        if method === :default
            return linmap_cholesky_default(to_matrix(A); perm=perm)
        elseif method === :autodiffable
            return linmap_cholesky_ldl_factorizations(sparse(to_matrix(A)); perm=perm)
        else
            throw(ArgumentError("Invalid factorization method: $method"))
        end
    end
    throw(ArgumentError("$A is not symmetric; Cholesky not available"))
end

function linmap_cholesky_default(A::AbstractMatrix; perm=nothing)
    if perm !== nothing
        return cholesky(A; perm=perm)
    end
    return cholesky(A)
end

function linmap_cholesky_ldl_factorizations(::AbstractMatrix; perm=nothing)
    error("LDLFactorizations.jl must be loaded to use it for factorization.")
end
