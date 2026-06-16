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

Optional fast paths (a generic fallback is provided):
- `selinv_dot(b, B)` — `dot(selinv, B) = tr(Q⁻¹ B)` without materializing selinv
"""
abstract type WorkspaceBackend end

# Generic fallback: materialize the selected inverse and dot. Backends that can
# contract against `B` without building the full `SparseMatrixCSC` (e.g.
# `CHOLMODBackend`) override this.
selinv_dot(b::WorkspaceBackend, B::AbstractMatrix) = dot(get_selinv(b), B)

"""
    CHOLMODBackend{T} <: WorkspaceBackend

CHOLMOD-based factorization backend. Owns a `CHOLMOD.Factor` whose symbolic
factorization (permutation, elimination tree, supernodes) is computed once
and reused across numeric refactorizations.

Also owns a persistent `CHOLMOD.Sparse` mirroring the precision's (fixed)
sparsity pattern. `refactorize!` copies the new precision values into this
buffer in place and refactorizes with `check = false`, avoiding the per-call
`Sparse(::Symmetric)` re-allocation and `check_sparse` structural validation
that `cholesky!(F, ::Symmetric)` would otherwise repeat on every Newton/grid
iteration.

Materializes the selected inverse lazily: callers needing only the diagonal
(marginal variances) skip building the full `SparseMatrixCSC`, and the full
matrix is built only when actually requested. Both caches are reset on
`refactorize!`.
"""
mutable struct CHOLMODBackend{T} <: WorkspaceBackend
    factor::SparseArrays.CHOLMOD.Factor{T}
    sparse::SparseArrays.CHOLMOD.Sparse{T}
    selinv_cache::Union{Nothing, SparseMatrixCSC{T, Int}}
    selinv_diag_cache::Union{Nothing, Vector{T}}
    # Supernodal selected inverse (the Takahashi recursion output, before any
    # `sparse()` materialization). It is cheap to read at a subset pattern, so the
    # `get_selinv` / `selinv_dot` / `selinv_extract_at` consumers all source from
    # it and the recursion runs at most once per refactorization.
    selinv_Z_cache::Union{Nothing, AbstractMatrix}
end

function CHOLMODBackend(Q::Symmetric{T, <:SparseMatrixCSC{T}}) where {T}
    return CHOLMODBackend{T}(
        cholesky(Q), SparseArrays.CHOLMOD.Sparse(Q), nothing, nothing, nothing
    )
end

"""
    _copy_sparse_values!(S::CHOLMOD.Sparse, A::SparseMatrixCSC) -> S

Copy `A`'s nonzero values into the value buffer of a persistent `CHOLMOD.Sparse`
in place. `A` must share `S`'s sparsity pattern (same `nnz` and CSC ordering),
which the workspace guarantees across refactorizations. This mirrors the
value-copy step of the `Sparse(::SparseMatrixCSC)` constructor (a straight
`unsafe_copyto!` into the `x` buffer) but skips both the re-allocation and the
`check_sparse` structural validation.
"""
function _copy_sparse_values!(S::SparseArrays.CHOLMOD.Sparse{T}, A::SparseMatrixCSC{T}) where {T}
    s = unsafe_load(pointer(S))
    n = nnz(A)
    Int(s.nzmax) == n || throw(
        ArgumentError(
            "CHOLMOD.Sparse buffer holds $(Int(s.nzmax)) values but A has $n nonzeros; " *
                "the sparsity pattern must be invariant across refactorizations."
        )
    )
    GC.@preserve S A unsafe_copyto!(Ptr{T}(s.x), pointer(nonzeros(A)), n)
    return S
end

function refactorize!(b::CHOLMODBackend, Q::Symmetric)
    # The sparsity pattern is fixed across refactorizations, so refresh the
    # persistent Sparse's values in place and refactorize with `check = false`
    # instead of letting `cholesky!(F, ::Symmetric)` rebuild a fresh Sparse and
    # re-run `check_sparse` every call. Bit-identical factor; ~10% less work.
    _copy_sparse_values!(b.sparse, Q.data)
    cholesky!(b.factor, b.sparse; check = false)
    b.selinv_cache = nothing
    b.selinv_diag_cache = nothing
    b.selinv_Z_cache = nothing
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

# The supernodal selected inverse `.Z` from the Takahashi recursion, cached so the
# full `get_selinv`, the `selinv_dot` contraction, and the `selinv_extract_at`
# subset-read all share a single recursion per refactorization.
function get_selinv_Z(b::CHOLMODBackend)
    # Bind to a local so the `=== nothing` check narrows the return type away from
    # `Nothing` (a direct `return b.selinv_Z_cache` would be `Union{Nothing,...}`,
    # which the downstream `sparse`/`dot`/`selinv_extract` calls cannot dispatch on).
    cache = b.selinv_Z_cache
    if cache === nothing
        cache = SelectedInversion.selinv(b.factor; depermute = true).Z
        b.selinv_Z_cache = cache
    end
    return cache
end

function get_selinv(b::CHOLMODBackend)
    if b.selinv_cache === nothing
        # `sparse` builds the result directly from the supernodal blocks. The
        # generic `SparseMatrixCSC` convert would instead call supernodal
        # `getindex` once per structural entry — orders of magnitude slower.
        b.selinv_cache = sparse(get_selinv_Z(b))
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

# `tr(Q⁻¹ B) = dot(selinv(Q), B)` for a `B` whose pattern is a subset of the
# Cholesky factor's. Dots straight against the supernodal selected-inverse blocks
# (merge-intersecting with `B`'s columns), skipping the full `SparseMatrixCSC`
# materialization that `get_selinv` would build — which dominates the cost when
# only this contraction is needed (the ForwardDiff `logdetcov` tangent). Accepts
# a `ForwardDiff.Dual`-valued `B` directly, accumulating a Dual result.
function selinv_dot(b::CHOLMODBackend, B::AbstractMatrix)
    return dot(get_selinv_Z(b), B)
end

# Read the selected inverse at `B`'s sparsity pattern (a `SparseMatrixCSC` with
# B's pattern, values = Σ there), without materializing the full selinv. On the
# CHOLMOD backend this streams straight from the supernodal blocks
# (`SelectedInversion.selinv_extract`); other backends fall back to extracting
# from the materialized selinv. Used by `diag(AΣAᵀ)` (predictor marginals), which
# needs only the observation-local pattern `≈ AᵀA`.
selinv_extract_at(b::WorkspaceBackend, B::SparseMatrixCSC) =
    SelectedInversion.selinv_extract(get_selinv(b), B)

selinv_extract_at(b::CHOLMODBackend, B::SparseMatrixCSC) =
    SelectedInversion.selinv_extract(get_selinv_Z(b), B)

function backend_backward_solve(b::CHOLMODBackend, x::AbstractVector)
    # CHOLMOD's FactorComponent \ only works with Vector, not SubArray views
    return b.factor.UP \ Vector(x)
end
