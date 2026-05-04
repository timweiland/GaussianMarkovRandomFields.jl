# `gaussian_approximation` IFT path for AutoDiffLikelihood with Dual hyperparams.
#
# Sequence (pure AD — no finite differences):
#   1. Strip Duals → primal AutoDiffLikelihood, run primal Newton.
#   2. Compute ∂(∇_x loglik)/∂θ at the converged primal x* via exact AD:
#      lift x* to `Dual{outer_tag, Float64, N}` with zero outer-partials,
#      call `loggrad` against the Dual-θ likelihood, read outer partials
#      off the result.
#   3. Solve Q_post · dx*/dθ_j = grad_dθ_j with the primal posterior factor.
#   4. Build x_star_dual carrying dx/dθ as outer partials.
#   5. Compute the total `dH/dθ = ∂H/∂θ + ∂H/∂x · dx/dθ` via exact AD: one
#      `loghessian` call on `x_star_dual` (with dx/dθ partials) + Dual-θ
#      likelihood. Read total derivative off the result's outer partials.
#   6. Assemble Dual `Q_post` by writing primal Q_prior values + H-derived
#      partials into a copy of Q_prior's exact sparse pattern.

"""
    _is_dual_autodifflik(lik) -> Bool

True when `lik::AutoDiffLikelihood` and any of its stored hyperparams
carries Dual partials.
"""
_is_dual_autodifflik(lik) = false
function _is_dual_autodifflik(lik::GMRFs.AutoDiffLikelihood)
    return GMRFs._hp_carries_ad_partials(lik.hyperparams)
end

# ----------------------------------------------------------------------------
# DI.hessian returns a dense Matrix even when the underlying Hessian is
# structurally diagonal (the typical case for a sum-of-pointwise loglik).
# Sniff for off-diagonal zeros and downcast so the IFT path can preserve
# Q_prior's sparse pattern through `_assemble_q_post_dual`. With a
# `pointwise_loglik_func` set, `loghessian` already returns a `Diagonal`
# directly via the main-src fast path, so this is a defensive fallback.
# ----------------------------------------------------------------------------

_maybe_downcast_diagonal(H::Diagonal) = H
_maybe_downcast_diagonal(H::SparseMatrixCSC) = H
function _maybe_downcast_diagonal(H::AbstractMatrix)
    n = size(H, 1)
    n == size(H, 2) || return H
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        iszero(H[i, j]) || return H
    end
    return Diagonal([H[i, i] for i in 1:n])
end

# ----------------------------------------------------------------------------
# Q_post_dual builder — preserves Q_prior's exact sparse structure for
# diagonal Hessians (the dominant case via pointwise_loglik_func). For
# subset-pattern sparse Hessians we still preserve Q_prior's pattern. For
# arbitrary dense Hessians we fall back to algebraic subtraction.
# ----------------------------------------------------------------------------

# Allocate `nzval_dual` as a copy of Q_prior.nzval lifted to `DualT` with
# zero partials everywhere. Both sparse-pattern-preserving variants below
# start from this and overwrite entries that intersect H.
function _alloc_dual_nzval_from_qprior(
        Q_prior::SparseMatrixCSC{Float64}, ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    PartialsT = ForwardDiff.Partials{N, Float64}
    zero_partials = PartialsT(ntuple(_ -> 0.0, Val(N)))
    nzval_dual = Vector{DualT}(undef, length(Q_prior.nzval))
    @inbounds for i in eachindex(Q_prior.nzval)
        nzval_dual[i] = DualT(Q_prior.nzval[i], zero_partials)
    end
    return nzval_dual
end

_q_post_with_pattern(Q_prior::SparseMatrixCSC, nzval_dual) =
    SparseMatrixCSC(Q_prior.m, Q_prior.n, copy(Q_prior.colptr), copy(Q_prior.rowval), nzval_dual)

# Diagonal H_dual — write Q_prior - H_dual into a copy of Q_prior's nzval.
# Reads primal H values via `ForwardDiff.value`, partials via
# `ForwardDiff.partials`. No FD anywhere.
function _assemble_q_post_dual(
        Q_prior::SparseMatrixCSC{Float64}, H_dual::Diagonal{<:ForwardDiff.Dual},
        ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    n = size(Q_prior, 1)
    nzval_dual = _alloc_dual_nzval_from_qprior(Q_prior, DualT, Val(N))
    @inbounds for j in 1:n
        for k in nzrange(Q_prior, j)
            if Q_prior.rowval[k] == j
                h = H_dual.diag[j]
                primal = Q_prior.nzval[k] - ForwardDiff.value(h)
                partials = ForwardDiff.Partials{N, Float64}(
                    ntuple(d -> -ForwardDiff.partials(h, d), Val(N))
                )
                nzval_dual[k] = DualT(primal, partials)
                break
            end
        end
    end
    return _q_post_with_pattern(Q_prior, nzval_dual)
end

# Sparse non-diagonal H_dual — match Q_prior's pattern, write H values where
# they overlap. Requires H's pattern to be a subset of Q_prior's; otherwise
# we'd silently drop H nonzeros outside Q_prior and return a wrong Q_post.
# Errors loudly in that case so the caller knows workspace reuse isn't
# applicable for this likelihood/prior combination.
function _assemble_q_post_dual(
        Q_prior::SparseMatrixCSC{Float64},
        H_dual::SparseMatrixCSC{<:ForwardDiff.Dual},
        ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    _check_h_pattern_subset(H_dual, Q_prior)
    n = size(Q_prior, 1)
    nzval_dual = _alloc_dual_nzval_from_qprior(Q_prior, DualT, Val(N))
    @inbounds for col in 1:n
        for k in nzrange(Q_prior, col)
            row = Q_prior.rowval[k]
            h = _sparse_lookup(H_dual, row, col)
            primal = Q_prior.nzval[k] - ForwardDiff.value(h)
            partials = ForwardDiff.Partials{N, Float64}(
                ntuple(d -> -ForwardDiff.partials(h, d), Val(N))
            )
            nzval_dual[k] = DualT(primal, partials)
        end
    end
    return _q_post_with_pattern(Q_prior, nzval_dual)
end

# Verify every nonzero of H is at a (row, col) that's also nonzero in
# Q_prior. If not, the IFT-with-workspace path can't preserve Q_prior's
# sparse pattern, and silently returning a Q_post built only from
# Q_prior's pattern would drop real H entries.
function _check_h_pattern_subset(H::SparseMatrixCSC, Q_prior::SparseMatrixCSC)
    for col in 1:size(H, 2)
        for k in nzrange(H, col)
            iszero(H.nzval[k]) && continue
            row = H.rowval[k]
            _sparse_lookup_present(Q_prior, row, col) || throw(
                ArgumentError(
                    "AutoDiffLikelihood IFT path: observation Hessian has a nonzero at " *
                        "($row, $col) outside the prior precision sparsity pattern. The " *
                        "workspace-reuse path requires H's pattern to be a subset of Q_prior. " *
                        "Use a likelihood whose Hessian is structurally diagonal " *
                        "(e.g. supply `pointwise_loglik_func`) or call `gaussian_approximation` " *
                        "without a `WorkspaceGMRF` prior."
                )
            )
        end
    end
    return nothing
end

function _sparse_lookup_present(A::SparseMatrixCSC, row::Int, col::Int)
    @inbounds for k in nzrange(A, col)
        A.rowval[k] == row && return true
    end
    return false
end

# Generic / dense H_dual — falls back to algebraic subtract. Result loses
# Q_prior's sparse structure, so workspace pattern checks downstream will
# fail. Only used as a last resort for non-pointwise + non-sparse
# Hessians, which is rare.
function _assemble_q_post_dual(
        Q_prior::SparseMatrixCSC{Float64},
        H_dual::AbstractMatrix{<:ForwardDiff.Dual},
        ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    n = size(Q_prior, 1)
    Q_post_dual = Matrix{DualT}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        h = H_dual[i, j]
        # Q_prior is Float64 sparse; entries outside its pattern are 0.
        q_primal = _sparse_lookup(Q_prior, i, j)
        primal = q_primal - ForwardDiff.value(h)
        partials = ForwardDiff.Partials{N, Float64}(
            ntuple(d -> -ForwardDiff.partials(h, d), Val(N))
        )
        Q_post_dual[i, j] = DualT(primal, partials)
    end
    return Q_post_dual
end

function _sparse_lookup(A::SparseMatrixCSC, row::Int, col::Int)
    @inbounds for k in nzrange(A, col)
        A.rowval[k] == row && return A.nzval[k]
    end
    return zero(eltype(A))
end

# ----------------------------------------------------------------------------
# Core IFT helper: works on a primal workspace prior + Dual-hp lik.
# ----------------------------------------------------------------------------

function _autodifflik_ift_workspace(
        prior_gmrf::GMRFs.WorkspaceGMRF{Float64},
        lik::GMRFs.AutoDiffLikelihood;
        ga_kwargs...
    )
    OuterTag, N = _outer_tag_and_npartials(lik.hyperparams)
    DualT = ForwardDiff.Dual{OuterTag, Float64, N}
    PartialsT = ForwardDiff.Partials{N, Float64}
    zero_partials = PartialsT(ntuple(_ -> 0.0, Val(N)))

    # Step 1: primal Newton on stripped likelihood.
    primal_lik = _primal_autodiff_likelihood(lik)
    posterior_primal = GMRFs.gaussian_approximation(prior_gmrf, primal_lik; ga_kwargs...)
    x_star = GMRFs.mean(posterior_primal)
    n = length(x_star)

    # Step 2: ∂(∇_x loglik)/∂θ at fixed x* via exact AD.
    # Lift x_star to Dual{OuterTag, Float64, N} with zero outer partials, so
    # the closure (which carries Dual θ) propagates θ-partials through arithmetic
    # while x's partial contribution is zero. The result's outer partials are
    # the partial-θ derivatives at fixed x.
    x_star_dual_zero = [DualT(x_star[i], zero_partials) for i in 1:n]
    g_dual_at_xstar = GMRFs.loggrad(x_star_dual_zero, lik)
    grad_dθ = ntuple(j -> [ForwardDiff.partials(g_dual_at_xstar[i], j) for i in 1:n], Val(N))

    # Step 3: IFT solve. We have neg_grad = Q_prior(x_star - μ) - loggrad,
    # which evaluates to 0 at the converged primal x_star (Newton condition).
    # Tangents: dx*/dθ_j = Q_post^{-1} · ∂(∇_x loglik)/∂θ_j (sign from IFT
    # cancels with neg_grad sign).
    ws = posterior_primal.workspace
    constrained = posterior_primal.constraints !== nothing
    ci = constrained ? posterior_primal.constraints : nothing
    dx = Matrix{Float64}(undef, n, N)
    for j in 1:N
        step = GMRFs.workspace_solve(ws, grad_dθ[j])
        if constrained
            A = ci.matrix
            step = step - ci.A_tilde_T * (ci.L_c \ (A * step))
        end
        dx[:, j] .= step
    end

    # Step 4: assemble Dual x* with dx/dθ as outer partials.
    x_star_dual = [
        DualT(x_star[i], PartialsT(ntuple(j -> dx[i, j], Val(N))))
            for i in 1:n
    ]

    # Step 5: total dH/dθ via one exact-AD loghessian call.
    # `loghessian(x_star_dual, lik)` propagates BOTH the x-tangent (via
    # x_star_dual's outer partials = dx/dθ) AND the θ-tangent (via lik's
    # Dual hyperparams), so the result's outer partials are the total
    # derivative `∂H/∂θ + ∂H/∂x · dx/dθ` — exactly what Q_post needs.
    H_dual = _maybe_downcast_diagonal(GMRFs.loghessian(x_star_dual, lik))
    Q_post_dual = _assemble_q_post_dual(prior_gmrf.precision, H_dual, DualT, Val(N))

    # Step 6: result. Diagonal/sparse Q_post_dual is structure-compatible with
    # the workspace pattern (we copied Q_prior's colptr/rowval), so the Dual
    # WorkspaceGMRF constructor accepts it.
    if prior_gmrf.constraints === nothing
        return GMRFs.WorkspaceGMRF(x_star_dual, Q_post_dual, ws)
    else
        ci_post = posterior_primal.constraints
        A_dense = ci_post.matrix
        log_AA_det = logdet(cholesky(Symmetric(A_dense * A_dense')))
        return _build_constrained_dual_workspace_gmrf(
            x_star_dual, Q_post_dual, ws,
            A_dense, ci_post.vector, ci_post.A_tilde_T, ci_post.L_c,
            log_AA_det, posterior_primal.version
        )
    end
end

# ----------------------------------------------------------------------------
# Dispatch hooks: route AutoDiffLikelihood-with-Dual-hp through the IFT.
# ----------------------------------------------------------------------------

function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.WorkspaceGMRF{Float64},
        obs_lik::GMRFs.AutoDiffLikelihood;
        kwargs...
    )
    if _is_dual_autodifflik(obs_lik)
        return _autodifflik_ift_workspace(prior_gmrf, obs_lik; kwargs...)
    end
    # Fall through to the primal-package's WorkspaceGMRF dispatch.
    return invoke(
        GMRFs.gaussian_approximation,
        Tuple{GMRFs.WorkspaceGMRF, GMRFs.ObservationLikelihood},
        prior_gmrf, obs_lik;
        kwargs...
    )
end
