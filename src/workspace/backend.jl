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
- `compute_selinv!(b)` — compute and cache selected inverse internally
- `get_selinv(b)` — return cached selected inverse (type is backend-specific)
- `get_selinv_diag(b)` — return diagonal of Q⁻¹ as Vector
- `backend_backward_solve(b, x)` — compute L^T \\ x for sampling
"""
abstract type WorkspaceBackend end

"""
    CHOLMODBackend{T} <: WorkspaceBackend

CHOLMOD-based factorization backend. Owns a `CHOLMOD.Factor` whose symbolic
factorization (permutation, elimination tree, supernodes) is computed once
and reused across numeric refactorizations.

Caches the selected inverse as a `SparseMatrixCSC` after `compute_selinv!`.
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
    b.selinv_cache = SelectedInversion.selinv(b.factor; depermute = true).Z
    b.selinv_diag_cache = diag(b.selinv_cache)
    return nothing
end

function get_selinv(b::CHOLMODBackend)
    return b.selinv_cache
end

function get_selinv_diag(b::CHOLMODBackend)
    return b.selinv_diag_cache
end

function backend_backward_solve(b::CHOLMODBackend, x::AbstractVector)
    # CHOLMOD's FactorComponent \ only works with Vector, not SubArray views
    return b.factor.UP \ Vector(x)
end
