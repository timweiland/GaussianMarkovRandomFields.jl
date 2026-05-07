export GMRFWorkspace, update_precision!, update_precision_values!,
    ensure_numeric!, ensure_selinv!, dimension

"""
    GMRFWorkspace{T, B}

A mutable workspace that persists a symbolic Cholesky factorization across
precision matrix updates with the same sparsity pattern. This is the key
optimization for INLA/TMB/HMC inner loops where the sparsity pattern is
fixed but numeric values change with hyperparameters.

The workspace owns an internal `Q` buffer (a `SparseMatrixCSC` whose `nzval`
is overwritten on updates) and a factorization backend that reuses its symbolic
analysis across numeric refactorizations. The backend also owns the selected
inverse cache in whatever format it prefers.

Multiple `WorkspaceGMRF`s can share the same workspace. A version counter
tracks whose data is currently loaded — if a `WorkspaceGMRF` finds the workspace
holds stale data, it transparently reloads and refactorizes.

# Fields
- `Q`: Internal precision matrix buffer (mutable `nzval`, fixed pattern)
- `backend`: Factorization backend (e.g., `CHOLMODBackend`, `CliqueTreesBackend`)
- `rhs`, `solution`: Preallocated work vectors
- `numeric_valid`, `selinv_valid`, `logdet_valid`: Lazy invalidation flags
- `logdet_cache`: Cached log-determinant
- `next_version`: Counter for assigning unique versions to `WorkspaceGMRF`s
- `loaded_version`: Version of the `WorkspaceGMRF` whose data is currently factorized
    (0 means data was loaded directly via `update_precision!`, not by a `WorkspaceGMRF`)
"""
mutable struct GMRFWorkspace{T <: Real, B <: WorkspaceBackend} <: AbstractLatentWorkspace
    Q::SparseMatrixCSC{T, Int}
    backend::B

    # Preallocated work vectors
    rhs::Vector{T}
    solution::Vector{T}

    # Lazy invalidation
    numeric_valid::Bool
    selinv_valid::Bool
    logdet_valid::Bool

    # Cached logdet (scalar — backend-independent, so lives here)
    logdet_cache::T

    # Version tracking for WorkspaceGMRF coherence
    next_version::Int
    loaded_version::Int
end

"""
    GMRFWorkspace(Q::SparseMatrixCSC{T}) where {T}

Create a workspace from a sparse SPD precision matrix using the default
CHOLMOD backend. Performs the initial symbolic + numeric Cholesky factorization.
Subsequent calls to `update_precision!` with matrices of the same sparsity
pattern will only redo the numeric phase.
"""
function GMRFWorkspace(Q::SparseMatrixCSC{T}) where {T}
    n = size(Q, 1)
    size(Q, 1) == size(Q, 2) || throw(ArgumentError("Q must be square"))

    backend = CHOLMODBackend(Symmetric(Q))

    return GMRFWorkspace{T, typeof(backend)}(
        copy(Q),
        backend,
        zeros(T, n),
        zeros(T, n),
        true,   # numeric_valid (just factorized)
        false,  # selinv_valid
        false,  # logdet_valid
        zero(T), # logdet_cache
        1,       # next_version
        0,       # loaded_version (no WorkspaceGMRF owns this yet)
    )
end

"""
    dimension(ws::GMRFWorkspace) -> Int

Return the dimension (number of rows/columns) of the precision matrix.
"""
dimension(ws::GMRFWorkspace) = size(ws.Q, 1)

"""
    _same_pattern(A::SparseMatrixCSC, B::SparseMatrixCSC) -> Bool

Check whether two sparse matrices have identical sparsity patterns.
"""
function _same_pattern(A::SparseMatrixCSC, B::SparseMatrixCSC)
    return size(A) == size(B) && A.colptr == B.colptr && A.rowval == B.rowval
end

"""
    _invalidate!(ws::GMRFWorkspace)

Mark all cached results as invalid. Called after precision values change.
"""
function _invalidate!(ws::GMRFWorkspace)
    ws.numeric_valid = false
    ws.selinv_valid = false
    ws.logdet_valid = false
    return nothing
end

"""
    update_precision!(ws::GMRFWorkspace, Q_new::SparseMatrixCSC)

Update the workspace's precision matrix values from `Q_new`. The sparsity
pattern of `Q_new` must exactly match the workspace's pattern (same `colptr`
and `rowval`). Only the numeric values (`nzval`) are copied.

Invalidates all cached results (factorization, selinv, logdet) and marks the
workspace as not owned by any `WorkspaceGMRF`.

!!! tip "Hot-loop performance"
    Each call runs an O(nnz) pattern-match check. In inner loops where
    pattern equality is guaranteed (e.g. pulling values from a fixed-pattern
    buffer), prefer [`update_precision_values!`](@ref), which takes the
    `nzval` vector directly and skips both the check and any intermediate
    `SparseMatrixCSC` construction.
"""
function update_precision!(ws::GMRFWorkspace, Q_new::SparseMatrixCSC)
    _same_pattern(ws.Q, Q_new) ||
        throw(
        ArgumentError(
            "Sparsity pattern mismatch: Q_new has different colptr/rowval. " *
                "GMRFWorkspace requires the same sparsity pattern across updates."
        )
    )
    copyto!(ws.Q.nzval, Q_new.nzval)
    _invalidate!(ws)
    ws.loaded_version = 0
    return nothing
end

"""
    update_precision_values!(ws::GMRFWorkspace, nzval::AbstractVector)

Update the workspace's precision matrix nonzero values directly.
`nzval` must have the same length as `ws.Q.nzval`.

Invalidates all cached results and marks the workspace as not owned by any
`WorkspaceGMRF`.
"""
function update_precision_values!(ws::GMRFWorkspace, nzval::AbstractVector)
    length(nzval) == length(ws.Q.nzval) ||
        throw(
        ArgumentError(
            "nzval length $(length(nzval)) does not match workspace Q nzval length $(length(ws.Q.nzval))"
        )
    )
    copyto!(ws.Q.nzval, nzval)
    _invalidate!(ws)
    ws.loaded_version = 0
    return nothing
end

"""
    ensure_numeric!(ws::GMRFWorkspace)

Ensure the numeric factorization is current. If the precision values have
been updated since the last factorization, performs a numeric-only
refactorization (reusing the existing symbolic analysis).
"""
function ensure_numeric!(ws::GMRFWorkspace)
    if !ws.numeric_valid
        refactorize!(ws.backend, Symmetric(ws.Q))
        ws.numeric_valid = true
        ws.selinv_valid = false
        ws.logdet_valid = false
    end
    return nothing
end

"""
    ensure_selinv!(ws::GMRFWorkspace)

Ensure the selected inverse is computed and cached in the backend.
Triggers numeric factorization if needed.
"""
function ensure_selinv!(ws::GMRFWorkspace)
    if !ws.selinv_valid
        ensure_numeric!(ws)
        compute_selinv!(ws.backend)
        ws.selinv_valid = true
    end
    return nothing
end

"""
    workspace_solve(ws::GMRFWorkspace, b::AbstractVector) -> Vector

Solve Q x = b using the workspace's factorization.
"""
function workspace_solve(ws::GMRFWorkspace, b::AbstractVector)
    ensure_numeric!(ws)
    return backend_solve(ws.backend, b)
end

"""
    LinearAlgebra.logdet(ws::GMRFWorkspace) -> Real

Return log|Q| using the workspace's factorization. Cached after first computation.
"""
function LinearAlgebra.logdet(ws::GMRFWorkspace)
    if !ws.logdet_valid
        ensure_numeric!(ws)
        ws.logdet_cache = compute_logdet(ws.backend)
        ws.logdet_valid = true
    end
    return ws.logdet_cache
end

"""
    logdet_cov(ws::GMRFWorkspace) -> Real

Return log|Q⁻¹| = -log|Q|.
"""
logdet_cov(ws::GMRFWorkspace) = -logdet(ws)

"""
    selinv(ws::GMRFWorkspace)

Return the selected inverse of Q — the values of `Q⁻¹` at the nonzero
positions of the Cholesky factor's sparsity pattern (which is a superset of
Q's pattern). This is *not* the full dense inverse; entries outside the
factor pattern are not computed.

Cached internally by the backend after first computation. Both built-in
backends (`CHOLMODBackend`, `CliqueTreesBackend`) return a `SparseMatrixCSC`;
the abstract interface allows backends to return other representations.
"""
function selinv(ws::GMRFWorkspace)
    ensure_selinv!(ws)
    return get_selinv(ws.backend)
end

"""
    selinv_diag(ws::GMRFWorkspace) -> Vector

Return the diagonal of Q⁻¹ via selected inversion.
"""
function selinv_diag(ws::GMRFWorkspace)
    ensure_selinv!(ws)
    return get_selinv_diag(ws.backend)
end

"""
    backward_solve(ws::GMRFWorkspace, x::AbstractVector) -> Vector

Compute L^T \\ x where Q = L L^T. Used for sampling from the GMRF.
"""
function backward_solve(ws::GMRFWorkspace, x::AbstractVector)
    ensure_numeric!(ws)
    return backend_backward_solve(ws.backend, x)
end
