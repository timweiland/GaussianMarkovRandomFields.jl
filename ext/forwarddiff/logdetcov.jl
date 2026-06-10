# Dual-valued logdetcov for `GMRF` and `WorkspaceGMRF`. Both use the same
# IFT trick: `logdet(Q)` is computed at primal values via the existing
# (Float64) factorization; tangents come from `-dot(selinv(Q), Q_dual)`,
# which Dual-arithmetic dispatches turn into the correct partials.

function logdetcov(x::GMRF{<:ForwardDiff.Dual})
    Qinv = GMRFs.selinv(x.linsolve_cache)
    primal = GMRFs.logdet_cov(x.linsolve_cache)
    tangent = -dot(Qinv, x.precision)
    return ForwardDiff.Dual{ForwardDiff.tagtype(tangent)}(primal, ForwardDiff.partials(tangent)...)
end

function logdetcov(x::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual})
    GMRFs.ensure_loaded!(x)
    primal = GMRFs.logdet_cov(x.workspace)
    # tr(Q⁻¹ Q̇) = dot(selinv(Q), Q_dual). `selinv_dot` contracts straight against
    # the supernodal selected-inverse blocks and accumulates a Dual, skipping the
    # full selinv `SparseMatrixCSC` materialization (the dominant cost here).
    tangent = -GMRFs.selinv_dot(x.workspace, x.precision)
    return ForwardDiff.Dual{ForwardDiff.tagtype(tangent)}(primal, ForwardDiff.partials(tangent)...)
end
