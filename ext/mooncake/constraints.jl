# Differentiable recomputation of the constraint quantities that
# `ConstraintInfo` / `ConstrainedGMRF` cache. Shared by the constrained
# `WorkspaceGMRF` and `ConstrainedGMRF` paths.

# Differentiable recomputation of the constraint Schur quantities that
# `ConstraintInfo`/`ConstrainedGMRF` cache: the precision enters through
# Ã = Q⁻¹Aᵀ, computed with `ldivwith` (the factor snapshot is only the
# solver, gradients flow to Q), and the small m×m Schur complement uses
# Mooncake's dense Cholesky rules. The corrections themselves come from the
# shared `_constraint_*` formulas.
"""
    _dense_constraints(A) -> Matrix

The constraint matrix as a dense array. Every differentiated `A * v` below goes
through this first: MooncakeSparse's pullback for a sparse `A * v` handles
square `A` only (`selupd_impl!` asserts it), while a constraint matrix is m×n
with m ≪ n. A constrained `WorkspaceGMRF` stores `A` sparse (the primal
constraint solves want it that way); `ConstrainedGMRF` already stores it dense,
and is passed through without a copy.
"""
_dense_constraints(A::Matrix) = A
_dense_constraints(A::AbstractMatrix) = Matrix(A)

# Returns the dense constraint matrix alongside the Schur quantities, so that
# the shared `_constraint_*` formulas are fed the dense copy.
function _mooncake_constraint_schur(Q::SparseMatrixCSC, F::ChordalCholesky, A::AbstractMatrix)
    A_dense = _dense_constraints(A)
    A_tilde_T = MooncakeSparse.ldivwith(Symmetric(Q), F, Matrix(A_dense'))
    S_c = cholesky(Symmetric(A_dense * A_tilde_T))
    return A_dense, A_tilde_T, S_c
end

function _ws_constraint_schur(d::WorkspaceGMRF, F::ChordalCholesky)
    return _mooncake_constraint_schur(d.precision, F, d.constraints.matrix)
end

function _ws_constraint_correction(d::WorkspaceGMRF, F::ChordalCholesky)
    A_dense, _, S_c = _ws_constraint_schur(d, F)
    return _constraint_log_correction(A_dense, d.constraints.vector, d.mean, S_c)
end
