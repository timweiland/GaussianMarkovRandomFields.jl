using SparseArrays

export WorkspacePool, checkout, checkin, with_workspace

"""
    WorkspacePool{T, B}

A thread-safe pool of `GMRFWorkspace` instances for parallel computation.

Uses a `Channel`-based checkout/checkin pattern (not `threadid()`-indexed,
since Julia tasks can migrate between threads). Each workspace in the pool
has its own CHOLMOD `Factor`, so concurrent factorizations don't interfere.

# Usage
```julia
pool = WorkspacePool(Q; size=Threads.nthreads())

# RAII pattern (recommended):
with_workspace(pool) do ws
    update_precision!(ws, Q_new)
    x = workspace_solve(ws, b)
end

# Manual checkout/checkin:
ws = checkout(pool)
try
    # use ws...
finally
    checkin(pool, ws)
end
```
"""
struct WorkspacePool{T, B}
    available::Channel{GMRFWorkspace{T, B}}
    size::Int
end

"""
    WorkspacePool(Q::SparseMatrixCSC{T}; size=Threads.nthreads())

Create a pool of `size` independent workspaces, each with its own symbolic
factorization of `Q`.
"""
function WorkspacePool(Q::SparseMatrixCSC{T}; size::Int = Threads.nthreads()) where {T}
    first_ws = GMRFWorkspace(Q)
    B = typeof(first_ws.backend)
    ch = Channel{GMRFWorkspace{T, B}}(size)
    put!(ch, first_ws)
    for _ in 2:size
        put!(ch, GMRFWorkspace(copy(Q)))
    end
    return WorkspacePool{T, B}(ch, size)
end

"""
    WorkspacePool(model::LatentModel; size=Threads.nthreads(), kwargs...)

Create a pool from a `LatentModel` at reference hyperparameters.
"""
function WorkspacePool(model::LatentModel; size::Int = Threads.nthreads(), kwargs...)
    Q = _ensure_sparse(precision_matrix(model; kwargs...))
    return WorkspacePool(Q; size = size)
end

"""
    checkout(pool::WorkspacePool) -> GMRFWorkspace

Take a workspace from the pool. Blocks if all workspaces are currently checked out.
"""
checkout(pool::WorkspacePool) = take!(pool.available)

"""
    checkin(pool::WorkspacePool, ws::GMRFWorkspace)

Return a workspace to the pool.
"""
checkin(pool::WorkspacePool, ws::GMRFWorkspace) = put!(pool.available, ws)

"""
    with_workspace(f, pool::WorkspacePool)

Execute `f(ws)` with a workspace checked out from the pool. The workspace is
guaranteed to be returned to the pool when `f` completes, even if it throws.

# Example
```julia
result = with_workspace(pool) do ws
    update_precision!(ws, Q_new)
    workspace_solve(ws, b)
end
```
"""
function with_workspace(f, pool::WorkspacePool)
    ws = checkout(pool)
    try
        return f(ws)
    finally
        checkin(pool, ws)
    end
end
