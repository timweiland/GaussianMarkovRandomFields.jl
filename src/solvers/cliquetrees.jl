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

_cliquetrees_factor(linsolve) = LinearSolve.@get_cacheval(linsolve, :CliqueTreesFactorization)

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
    L = sparse(triangular(Y))
    invp = collect(Y.invp)
    Σ_perm = L + L' - Diagonal(diag(L))
    return Σ_perm[invp, invp]
end
