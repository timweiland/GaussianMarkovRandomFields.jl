using SparseArrays

export make_workspace, make_workspace_pool

"""
    make_workspace(m::LatentModel; θ_ref...) -> AbstractLatentWorkspace

Construct a workspace for evaluating `m` repeatedly across hyperparameter
values. The returned workspace is suitable for use as the second argument
to `m(ws; θ...)`.

Default implementation: returns `GMRFWorkspace(m; θ_ref...)`. Downstream
`LatentModel` subtypes may override `make_workspace` to return a peer
workspace type when their factorization structure doesn't fit
`GMRFWorkspace`'s "one sparse precision, one factorization" shape.

!!! warning "Experimental API (pre v0.5)"
    Surface may shift until validated against at least two downstream
    implementations.

# Example
```julia
model = AR1Model(100)
ws = make_workspace(model; τ = 1.0, ρ = 0.5)
# loop:
for θ in θ_grid
    gmrf = model(ws; θ...)
    logpdf(gmrf, z)
end
```
"""
make_workspace(m::LatentModel; kwargs...) = GMRFWorkspace(m; kwargs...)

"""
    make_workspace_pool(m::LatentModel; size=Threads.nthreads(), θ_ref...) -> AbstractLatentWorkspacePool

Construct a task-safe pool of workspaces for parallel evaluation of `m`.

Default implementation: returns `WorkspacePool(m; size=size, θ_ref...)`.
Downstream `LatentModel` subtypes may override to return a peer pool type.

!!! warning "Experimental API (pre v0.5)"
    Surface may shift until validated against at least two downstream
    implementations.
"""
make_workspace_pool(m::LatentModel; size::Int = Threads.nthreads(), kwargs...) =
    WorkspacePool(m; size = size, kwargs...)

"""
    GMRFWorkspace(model::LatentModel; kwargs...)

Create a `GMRFWorkspace` from a `LatentModel` at reference hyperparameter values.
The precision matrix is computed at the given hyperparameters, converted to sparse
if needed, and used to initialize the workspace (performing the one-time symbolic
factorization).

# Example
```julia
model = AR1Model(100)
ws = GMRFWorkspace(model; τ=1.0, ρ=0.5)  # one-time symbolic factorization
```
"""
function GMRFWorkspace(model::LatentModel; kwargs...)
    Q = precision_matrix(model; kwargs...)
    Q_sparse = _ensure_sparse(Q)
    return GMRFWorkspace(Q_sparse)
end

"""
    GMRFWorkspace(model::LatentModel, obs_lik::ObservationLikelihood; kwargs...)

Create a `GMRFWorkspace` with a joint sparsity pattern that includes both the
prior precision pattern and the observation Hessian pattern. This is needed when
the observation Hessian is non-diagonal (e.g., with a design matrix A).

The workspace is initialized at the given reference hyperparameters. The observation
Hessian pattern is detected by evaluating `loghessian` at a dummy point.
"""
function GMRFWorkspace(model::LatentModel, obs_lik::ObservationLikelihood; kwargs...)
    Q_prior = _ensure_sparse(precision_matrix(model; kwargs...))
    n = size(Q_prior, 1)

    # Evaluate Hessian at zero to get the sparsity pattern
    x_dummy = zeros(n)
    H = loghessian(x_dummy, obs_lik)
    H_sparse = _ensure_sparse_hessian(H, n)

    # Joint pattern: Q_prior + |H| (using absolute values to get the union pattern)
    # Then copy Q_prior's actual values into the joint pattern
    Q_joint_pattern = Q_prior + _abs_pattern(H_sparse)
    # Reset values to Q_prior (the observation entries become zero)
    _copy_values_into!(Q_joint_pattern, Q_prior)

    return GMRFWorkspace(Q_joint_pattern)
end

"""
    (model::LatentModel)(ws::GMRFWorkspace; kwargs...)

Create a `WorkspaceGMRF` from a `LatentModel` using an existing workspace.
Computes fresh mean and precision from hyperparameters, updates the workspace,
and returns a workspace-backed GMRF (with constraints embedded if the model has them).

# Example
```julia
model = AR1Model(100)
ws = GMRFWorkspace(model; τ=1.0, ρ=0.5)

prior = model(ws; τ=2.0, ρ=0.3)  # WorkspaceGMRF, numeric-only refactorization
```
"""
function (model::LatentModel)(ws::GMRFWorkspace; kwargs...)
    μ = mean(model; kwargs...)
    Q = precision_matrix(model; kwargs...)
    Q_sparse = _ensure_sparse(Q)
    constraint_info = constraints(model; kwargs...)

    # Pad Q's values into the workspace's sparsity pattern with zeros at
    # observation-Hessian-only positions. Allows the joint-pattern workspace
    # built by `GMRFWorkspace(model, obs_lik)` to accept prior-pattern Q's
    # from the model. A no-op if patterns already match.
    Q_for_ws = _pad_to_workspace_pattern(Q_sparse, ws)

    update_precision!(ws, Q_for_ws)

    if constraint_info === nothing
        return WorkspaceGMRF(μ, Q_for_ws, ws)
    else
        A, e = constraint_info
        return WorkspaceGMRF(μ, Q_for_ws, ws, A, e)
    end
end

# --- Helpers ---
# Note: _ensure_sparse(::SparseMatrixCSC) and _ensure_sparse(::AbstractMatrix)
# are defined in latent_models/combined.jl. We only add the 2-arg overloads here.

"""Convert a Hessian (possibly Diagonal) to sparse with explicit dimension."""
_ensure_sparse_hessian(H::Diagonal, n::Int) = sparse(H)
_ensure_sparse_hessian(H::SparseMatrixCSC, ::Int) = H
_ensure_sparse_hessian(H::AbstractMatrix, ::Int) = sparse(H)

"""Create a sparse matrix with absolute values of nonzeros (for pattern union)."""
function _abs_pattern(H::SparseMatrixCSC{T}) where {T}
    return SparseMatrixCSC(H.m, H.n, H.colptr, H.rowval, abs.(H.nzval))
end

"""
    _pad_to_workspace_pattern(Q::SparseMatrixCSC, ws::GMRFWorkspace) -> SparseMatrixCSC

Return `Q` padded into `ws.Q`'s sparsity pattern, with zeros at positions
present in `ws.Q` but absent from `Q`. Returns `Q` itself if patterns
already match. Throws `ArgumentError` if `Q` has nonzeros outside `ws.Q`'s
pattern (which would silently lose data).

Used by the `(::LatentModel)(ws; θ...)` fast path so that workspaces
built with the joint prior + observation-Hessian pattern can still accept
prior-pattern precision matrices from the model.
"""
function _pad_to_workspace_pattern(Q::SparseMatrixCSC, ws::GMRFWorkspace)
    _same_pattern(Q, ws.Q) && return Q

    T = eltype(Q)
    Q_padded = SparseMatrixCSC(
        ws.Q.m, ws.Q.n,
        copy(ws.Q.colptr), copy(ws.Q.rowval),
        zeros(T, length(ws.Q.nzval))
    )

    Q_rows = rowvals(Q)
    pad_rows = rowvals(Q_padded)
    @inbounds for col in 1:size(Q, 2)
        pad_ptr = first(nzrange(Q_padded, col))
        pad_end = last(nzrange(Q_padded, col))
        for src_idx in nzrange(Q, col)
            src_row = Q_rows[src_idx]
            while pad_ptr <= pad_end && pad_rows[pad_ptr] < src_row
                pad_ptr += 1
            end
            if pad_ptr > pad_end || pad_rows[pad_ptr] != src_row
                throw(
                    ArgumentError(
                        "Q has nonzero at ($src_row, $col) outside the workspace pattern."
                    )
                )
            end
            Q_padded.nzval[pad_ptr] = Q.nzval[src_idx]
        end
    end
    return Q_padded
end

"""Copy values from `src` into `dst` at matching sparsity positions.
Positions in `dst` not present in `src` are set to zero."""
function _copy_values_into!(dst::SparseMatrixCSC, src::SparseMatrixCSC)
    fill!(dst.nzval, 0.0)
    dst_rows = rowvals(dst)
    src_rows = rowvals(src)
    for col in 1:size(dst, 2)
        dst_ptr = first(nzrange(dst, col))
        dst_end = last(nzrange(dst, col))
        for src_idx in nzrange(src, col)
            src_row = src_rows[src_idx]
            while dst_ptr <= dst_end && dst_rows[dst_ptr] < src_row
                dst_ptr += 1
            end
            if dst_ptr <= dst_end && dst_rows[dst_ptr] == src_row
                dst.nzval[dst_ptr] = src.nzval[src_idx]
            end
        end
    end
    return nothing
end
