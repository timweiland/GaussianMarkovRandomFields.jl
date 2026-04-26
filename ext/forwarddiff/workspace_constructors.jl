# WorkspaceGMRF ForwardDiff support.
#
# When Dual numbers flow into WorkspaceGMRF construction, create the workspace
# from primal values (CHOLMOD can't handle Duals) while preserving Duals in
# the mean and precision fields for tangent propagation.

# update_precision! with a Dual-valued Q: strip to primal values before forwarding.
# This unblocks the LatentModel(ws; θ::Dual...) reuse path, which calls
# update_precision! with a Dual Q produced from Dual hyperparameters.
function GMRFs.update_precision!(
        ws::GMRFs.GMRFWorkspace, Q::SparseMatrixCSC{<:ForwardDiff.Dual}
    )
    Q_primal = SparseMatrixCSC(
        Q.m, Q.n, Q.colptr, Q.rowval, ForwardDiff.value.(Q.nzval)
    )
    return GMRFs.update_precision!(ws, Q_primal)
end

function _construct_forwarddiff_workspace_gmrf(
        mean::AbstractVector, Q::SparseMatrixCSC
    )
    T = promote_type(eltype(mean), eltype(Q))
    mean_T = eltype(mean) === T ? mean : convert(AbstractVector{T}, mean)
    Q_T = eltype(Q) === T ? Q : SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, convert(Vector{T}, Q.nzval))

    # Create workspace from primal values
    Q_primal = SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, ForwardDiff.value.(Q.nzval))
    ws = GMRFs.GMRFWorkspace(Q_primal)

    version = GMRFs._next_version!(ws)
    ws.loaded_version = version
    return GMRFs.WorkspaceGMRF{T, typeof(ws.backend), typeof(ws), Nothing}(
        Vector{T}(mean_T), copy(Q_T), ws, nothing, version
    )
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC
    )
    return _construct_forwarddiff_workspace_gmrf(mean, Q)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector,
        Q::SparseMatrixCSC{<:ForwardDiff.Dual}
    )
    return _construct_forwarddiff_workspace_gmrf(mean, Q)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC{<:ForwardDiff.Dual}
    )
    return _construct_forwarddiff_workspace_gmrf(mean, Q)
end

# 3-arg constructor with an existing workspace. The workspace stays primal;
# the WorkspaceGMRF holds Dual mean/precision for tangent propagation.
# Loading into the workspace happens lazily via the Dual `ensure_loaded!`
# override below (which strips to primal).
function _construct_forwarddiff_workspace_gmrf_with_ws(
        mean::AbstractVector, Q::SparseMatrixCSC, ws::GMRFs.GMRFWorkspace
    )
    GMRFs._check_workspace_pattern(Q, ws)
    T = promote_type(eltype(mean), eltype(Q))
    mean_T = eltype(mean) === T ? mean : convert(AbstractVector{T}, mean)
    Q_T = if eltype(Q) === T
        Q
    else
        SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, convert(Vector{T}, Q.nzval))
    end
    version = GMRFs._next_version!(ws)
    return GMRFs.WorkspaceGMRF{T, typeof(ws.backend), typeof(ws), Nothing}(
        Vector{T}(mean_T), copy(Q_T), ws, nothing, version
    )
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC,
        ws::GMRFs.GMRFWorkspace
    )
    return _construct_forwarddiff_workspace_gmrf_with_ws(mean, Q, ws)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector,
        Q::SparseMatrixCSC{<:ForwardDiff.Dual},
        ws::GMRFs.GMRFWorkspace
    )
    return _construct_forwarddiff_workspace_gmrf_with_ws(mean, Q, ws)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC{<:ForwardDiff.Dual},
        ws::GMRFs.GMRFWorkspace
    )
    return _construct_forwarddiff_workspace_gmrf_with_ws(mean, Q, ws)
end

# Dual WorkspaceGMRFs hold Dual-valued precision but the workspace buffer is
# Float64. Extract primal values for reloading so version coherence works.
function GMRFs.ensure_loaded!(d::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual})
    ws = d.workspace
    if ws.loaded_version != d.version
        copyto!(ws.Q.nzval, ForwardDiff.value.(d.precision.nzval))
        GMRFs._invalidate!(ws)
        ws.loaded_version = d.version
    end
    return nothing
end

# logdetcov for Dual-valued WorkspaceGMRF: same approach as GMRF{Dual}
function logdetcov(x::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual})
    GMRFs.ensure_loaded!(x)
    Qinv = GMRFs.selinv(x.workspace)
    primal = GMRFs.logdet_cov(x.workspace)
    # dot(Qinv, Q_dual) naturally produces a Dual via ForwardDiff overloads
    tangent = -dot(Qinv, x.precision)
    return ForwardDiff.Dual{ForwardDiff.tagtype(tangent)}(primal, ForwardDiff.partials(tangent)...)
end
