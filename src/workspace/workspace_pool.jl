using SparseArrays

export WorkspacePool, checkout, checkin, with_workspace

"""
    WorkspacePool{T, B}

A task-safe pool of `GMRFWorkspace` instances for use across multiple Julia
tasks. Uses a `Channel`-based checkout/checkin pattern (not `threadid()`-indexed,
since Julia tasks can migrate between threads).

Each workspace in the pool owns its own backend factorization, so
checkout/checkin is the right primitive for sharing across concurrent tasks.

!!! note "Concurrent factorization"
    Whether tasks can actually factorize in parallel depends on the backend.
    `CHOLMODBackend` (the default) goes through CHOLMOD, which holds a global
    lock — concurrent factorizations serialize gracefully but do not run in
    parallel. `CliqueTreesBackend` is pure-Julia and thread-safe, so a pool of
    `CliqueTreesBackend` workspaces *does* parallelize numeric factorization.
    Build the pool with the desired backend if parallel factorization matters.

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
struct WorkspacePool{T, B} <: AbstractLatentWorkspacePool
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
    with_workspace(f, pool::AbstractLatentWorkspacePool)

Execute `f(ws)` with a workspace checked out from the pool. The workspace is
guaranteed to be returned to the pool when `f` completes, even if it throws.

This is the protocol-level default: any `AbstractLatentWorkspacePool` subtype
that implements [`checkout`](@ref) and [`checkin`](@ref) inherits this
behavior automatically. Subtypes may override to add logging, metrics, or
alternate resource semantics.

# Example
```julia
result = with_workspace(pool) do ws
    update_precision!(ws, Q_new)
    workspace_solve(ws, b)
end
```
"""
function with_workspace(f, pool::AbstractLatentWorkspacePool)
    ws = checkout(pool)
    try
        return f(ws)
    finally
        checkin(pool, ws)
    end
end
