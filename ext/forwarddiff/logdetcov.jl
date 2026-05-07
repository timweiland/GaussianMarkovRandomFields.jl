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
    Qinv = GMRFs.selinv(x.workspace)
    primal = GMRFs.logdet_cov(x.workspace)
    # dot(Qinv, Q_dual) naturally produces a Dual via ForwardDiff overloads
    tangent = -dot(Qinv, x.precision)
    return ForwardDiff.Dual{ForwardDiff.tagtype(tangent)}(primal, ForwardDiff.partials(tangent)...)
end
