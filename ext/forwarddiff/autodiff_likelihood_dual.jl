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

# `loggrad` / `loghessian` overrides for closure-Dual `AutoDiffLikelihood`.
# Two problems with going through the default DI path:
#
#   1. DI's prep cache is keyed on `eltype(x)`, which doesn't reflect
#      closure-Dual outputs. The prepared output buffer ends up
#      undersized and the gradient comes back as `Vector{Any}`.
#   2. Calling `ForwardDiff.gradient(f, x::Vector{Float64})` directly
#      doesn't help: ForwardDiff allocates a fresh inner tag for the
#      gradient pass that has no defined ordering relative to the
#      closure's outer Dual tag — `DualMismatchError` at the first
#      arithmetic op that mixes the two.
#
# Lift `x` to `Vector{OutT}` first so the inner FD tag wraps the outer
# (closure-captured) layer instead of running parallel to it. The
# resulting Dual nesting is unambiguous and FD's standard arithmetic
# handles it.
function GMRFs.loggrad(
        x,
        obs_lik::GMRFs.AutoDiffLikelihood{F, B, SB, PF, OutT}
    ) where {F, B, SB, PF, OutT <: ForwardDiff.Dual}
    x_lifted = convert(Vector{OutT}, x)
    return ForwardDiff.gradient(obs_lik.loglik_func, x_lifted)
end

function GMRFs.loghessian(
        x,
        obs_lik::GMRFs.AutoDiffLikelihood{F, B, SB, PF, OutT}
    ) where {F, B, SB, PF, OutT <: ForwardDiff.Dual}
    x_lifted = convert(Vector{OutT}, x)
    if obs_lik.pointwise_loglik_func !== nothing
        return GMRFs._pointwise_diagonal_hessian(obs_lik.pointwise_loglik_func, x_lifted)
    end
    return ForwardDiff.hessian(obs_lik.loglik_func, x_lifted)
end

# Element-wise primal stripping that preserves matrix structure (Diagonal
# stays Diagonal, sparse stays sparse, dense stays dense). Broadcasting
# `ForwardDiff.value` over a `Diagonal` materializes a dense `Matrix`,
# which would defeat `_subtract_diagonal_hessian!`'s structure dispatch.
_strip_dual(H::Diagonal) = Diagonal(ForwardDiff.value.(H.diag))
function _strip_dual(H::SparseMatrixCSC)
    return SparseMatrixCSC(H.m, H.n, H.colptr, H.rowval, ForwardDiff.value.(H.nzval))
end
_strip_dual(H::AbstractMatrix) = ForwardDiff.value.(H)
