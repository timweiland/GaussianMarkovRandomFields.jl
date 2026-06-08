using LinearAlgebra
using SparseArrays
using SelectedInversion

export WorkspaceBackend, CHOLMODBackend

"""
    WorkspaceBackend

Abstract type for factorization backends used by `GMRFWorkspace`.

Each backend must implement:
- `refactorize!(b, Q::Symmetric)` — numeric-only refactorization reusing symbolic
- `backend_solve(b, rhs)` — solve Q x = rhs
- `compute_logdet(b)` — log determinant of Q
- `compute_selinv!(b)` — prepare the selected inverse caches (may be lazy)
- `get_selinv(b)` — return the selected inverse (type is backend-specific)
- `get_selinv_diag(b)` — return diagonal of Q⁻¹ as Vector
- `backend_backward_solve(b, x)` — compute L^T \\ x for sampling
"""
abstract type WorkspaceBackend end

"""
    CHOLMODBackend{T} <: WorkspaceBackend

CHOLMOD-based factorization backend. Owns a `CHOLMOD.Factor` whose symbolic
factorization (permutation, elimination tree, supernodes) is computed once
and reused across numeric refactorizations.

Materializes the selected inverse lazily: callers needing only the diagonal
(marginal variances) skip building the full `SparseMatrixCSC`, and the full
matrix is built only when actually requested. Both caches are reset on
`refactorize!`.
"""
mutable struct CHOLMODBackend{T} <: WorkspaceBackend
    factor::SparseArrays.CHOLMOD.Factor{T}
    selinv_cache::Union{Nothing, SparseMatrixCSC{T, Int}}
    selinv_diag_cache::Union{Nothing, Vector{T}}
end

function CHOLMODBackend(Q::Symmetric{T, <:SparseMatrixCSC{T}}) where {T}
    return CHOLMODBackend{T}(cholesky(Q), nothing, nothing)
end

function refactorize!(b::CHOLMODBackend, Q::Symmetric)
    cholesky!(b.factor, Q)
    b.selinv_cache = nothing
    b.selinv_diag_cache = nothing
    return nothing
end

function backend_solve(b::CHOLMODBackend, rhs::AbstractVector)
    return b.factor \ rhs
end

function compute_logdet(b::CHOLMODBackend)
    return logdet(b.factor)
end

function compute_selinv!(b::CHOLMODBackend)
    # The full sparse selected inverse and its diagonal are materialized lazily
    # by the getters below. The `var` path needs only the diagonal while the
    # autodiff path needs only the full matrix, so eagerly computing both here
    # wasted work on every factorization. Caches are reset in `refactorize!`.
    return nothing
end

function get_selinv(b::CHOLMODBackend)
    if b.selinv_cache === nothing
        # `sparse` builds the result directly from the supernodal blocks. The
        # generic `SparseMatrixCSC` convert would instead call supernodal
        # `getindex` once per structural entry — orders of magnitude slower.
        b.selinv_cache = sparse(SelectedInversion.selinv(b.factor; depermute = true).Z)
    end
    return b.selinv_cache
end

function get_selinv_diag(b::CHOLMODBackend)
    if b.selinv_diag_cache === nothing
        # Reuse the full selected inverse if it is already materialized;
        # otherwise compute just the diagonal, skipping the sparse build.
        b.selinv_diag_cache = b.selinv_cache === nothing ?
            SelectedInversion.selinv_diag(b.factor) :
            diag(b.selinv_cache)
    end
    return b.selinv_diag_cache
end

function backend_backward_solve(b::CHOLMODBackend, x::AbstractVector)
    # CHOLMOD's FactorComponent \ only works with Vector, not SubArray views
    return b.factor.UP \ Vector(x)
end
