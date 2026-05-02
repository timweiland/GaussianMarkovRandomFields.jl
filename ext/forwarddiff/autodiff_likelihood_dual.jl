# Strip-on-eval view used during the primal Newton iteration when an
# `AutoDiffLikelihood`'s closure has captured Duals (detected via the
# probed `OutT` type parameter on the struct). The wrapper forwards
# `loglik` / `loggrad` / `loghessian` to the inner likelihood and applies
# `ForwardDiff.value` to the result, so the Newton loop sees pure Float64
# values and `_subtract_diagonal_hessian!` doesn't try to assign Duals
# into the workspace's Float64 `Q.nzval` buffer.
#
# Tangent propagation happens after Newton converges, via the existing
# IFT helpers in `gmrf_gaussian_approximation.jl` /
# `workspace_gaussian_approximation.jl` — those evaluate the Dual
# obs_lik directly at `x*`.

struct _PrimalAutoDiffView{L <: GMRFs.AutoDiffLikelihood} <: GMRFs.ObservationLikelihood
    inner::L
end

function _primal_obs_lik(lik::_DualAutoDiffLik)
    return _PrimalAutoDiffView(lik)
end

GMRFs.loglik(x, view::_PrimalAutoDiffView) = ForwardDiff.value(GMRFs.loglik(x, view.inner))

function GMRFs.loggrad(x, view::_PrimalAutoDiffView)
    g = GMRFs.loggrad(x, view.inner)
    return ForwardDiff.value.(g)
end

GMRFs.loghessian(x, view::_PrimalAutoDiffView) = _strip_dual(GMRFs.loghessian(x, view.inner))

# Element-wise primal stripping that preserves matrix structure (Diagonal
# stays Diagonal, sparse stays sparse, dense stays dense). Broadcasting
# `ForwardDiff.value` over a `Diagonal` materializes a dense `Matrix`,
# which would defeat `_subtract_diagonal_hessian!`'s structure dispatch.
_strip_dual(H::Diagonal) = Diagonal(ForwardDiff.value.(H.diag))
function _strip_dual(H::SparseMatrixCSC)
    return SparseMatrixCSC(H.m, H.n, H.colptr, H.rowval, ForwardDiff.value.(H.nzval))
end
_strip_dual(H::AbstractMatrix) = ForwardDiff.value.(H)
