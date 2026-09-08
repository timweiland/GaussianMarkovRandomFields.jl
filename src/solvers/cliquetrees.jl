using LinearSolve
using LinearAlgebra
using SparseArrays
using CliqueTrees.Multifrontal: ChordalCholesky, selinv!, triangular

# GMRF backend based on CliqueTrees.jl's multifrontal Cholesky (pure Julia).
# LinearSolve ships the algorithm (`CliqueTreesFactorization`) and its `solve!`
# implementation; this file wires the GMRF-specific operations — logdet,
# selected inversion, and backward solve — to the `ChordalCholesky` cacheval.

supports_selinv(::LinearSolve.CliqueTreesFactorization) = Val{true}()
supports_backward_solve(::LinearSolve.CliqueTreesFactorization) = Val{true}()

# The `::ChordalCholesky` assertion is what keeps this inferrable: `@get_cacheval`
# returns `Any`, so without it `selinv!` below union-splits across every method
# it has — including one returning an `Integer`, whose branch has no
# `_chordal_selinv_diag`/`_chordal_selinv_full` method and which JET's error
# analysis flags (on Julia 1.11+; 1.10's inference does not reach it). The
# cacheval for this algorithm is always a `ChordalCholesky`, so the assertion
# can only fail if LinearSolve changes what it stores — loudly, which is what
# we would want.
_cliquetrees_factor(linsolve) =
    LinearSolve.@get_cacheval(linsolve, :CliqueTreesFactorization)::ChordalCholesky

function _logdet_cov_impl(linsolve, ::LinearSolve.CliqueTreesFactorization)
    return -logdet(_cliquetrees_factor(linsolve))
end

# `selinv!` overwrites the factor in place, so both entry points run on a
# scratch copy and leave the solve factorization intact.
function _selinv_diag_impl(linsolve, ::LinearSolve.CliqueTreesFactorization)
    return _chordal_selinv_diag(selinv!(copy(_cliquetrees_factor(linsolve))))
end

function _selinv_impl(linsolve, ::LinearSolve.CliqueTreesFactorization)
    return Symmetric(_chordal_selinv_full(selinv!(copy(_cliquetrees_factor(linsolve)))))
end

function _backward_solve_impl(linsolve, x, ::LinearSolve.CliqueTreesFactorization)
    F = _cliquetrees_factor(linsolve)
    return F.P \ (F.U \ x)
end

# --- Helpers over a selinv!-overwritten ChordalCholesky ---
# (shared with the workspace CliqueTreesBackend)

"""
    _chordal_selinv_diag(Y::ChordalCholesky) -> Vector

Diagonal of the selected inverse stored in `Y` (a factor overwritten by
`selinv!`), in the original (unpermuted) ordering.
"""
_chordal_selinv_diag(Y::ChordalCholesky) = diag(triangular(Y))[collect(Y.invp)]

"""
    _chordal_selinv_full(Y::ChordalCholesky) -> SparseMatrixCSC

Selected inverse stored in `Y` (a factor overwritten by `selinv!`) as a
sparse matrix with both triangles of the factor's fill pattern, in the
original (unpermuted) ordering.
"""
function _chordal_selinv_full(Y::ChordalCholesky)
    # Symmetrize and unpermute in a single COO build. The transparent spelling —
    # `Σ = L + L' - Diagonal(diag(L))` followed by `Σ[invp, invp]` — costs three
    # sparse matrix constructions and a two-vector `getindex`, and dominated the
    # whole selected inversion (~3x the Takahashi recursion it post-processes).
    # Emitting both symmetric positions of each stored entry directly at their
    # permuted indices is ~3x faster and gives the identical matrix.
    L = sparse(triangular(Y))
    perm = Y.perm
    n = size(L, 1)
    rows = rowvals(L)
    vals = nonzeros(L)
    upper = 2 * nnz(L)
    rows_out = Vector{Int}(undef, upper)
    cols_out = Vector{Int}(undef, upper)
    vals_out = Vector{eltype(vals)}(undef, upper)
    k = 0
    @inbounds for j in 1:n
        pj = perm[j]
        for idx in nzrange(L, j)
            i = rows[idx]
            v = vals[idx]
            pi = perm[i]
            k += 1
            rows_out[k] = pi
            cols_out[k] = pj
            vals_out[k] = v
            # `triangular` is lower-triangular, so off-diagonal entries are the
            # only ones with a distinct mirror position.
            if i != j
                k += 1
                rows_out[k] = pj
                cols_out[k] = pi
                vals_out[k] = v
            end
        end
    end
    return sparse(
        resize!(rows_out, k), resize!(cols_out, k), resize!(vals_out, k), n, n
    )
end
