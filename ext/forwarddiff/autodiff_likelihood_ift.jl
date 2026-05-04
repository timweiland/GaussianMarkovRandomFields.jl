# `gaussian_approximation` IFT path for AutoDiffLikelihood with Dual hyperparams.
#
# Sequence:
#   1. Strip Duals → primal AutoDiffLikelihood, run primal Newton.
#   2. For each outer-FD partial direction j, compute ∂(∇_x loglik)/∂θ_j at
#      the converged primal x* via central finite differences in θ-space
#      (no nested AD).
#   3. Solve Q_post · dx*/dθ_j = grad_dθ_j with the primal posterior factor.
#   4. Assemble Dual x* and Dual Q_post; return as the appropriate result type
#      for the prior (GMRF / ConstrainedGMRF / WorkspaceGMRF).

"""
    _is_dual_autodifflik(lik) -> Bool

True when `lik::AutoDiffLikelihood` and any of its stored hyperparams
carries Dual partials.
"""
_is_dual_autodifflik(lik) = false
function _is_dual_autodifflik(lik::GMRFs.AutoDiffLikelihood)
    return GMRFs._hp_carries_dual(lik.hyperparams)
end

# ----------------------------------------------------------------------------
# Per-direction θ-tangent of `loggrad` at fixed x via central FD on θ.
# ----------------------------------------------------------------------------

function _grad_dθ_central(lik::GMRFs.AutoDiffLikelihood, x_star, j::Int, ε::Float64)
    lik_plus = _perturbed_autodiff_likelihood(lik, j, +ε)
    lik_minus = _perturbed_autodiff_likelihood(lik, j, -ε)
    return (GMRFs.loggrad(x_star, lik_plus) .- GMRFs.loggrad(x_star, lik_minus)) ./ (2ε)
end

function _hess_dθ_central(lik::GMRFs.AutoDiffLikelihood, x_star, dx_j, j::Int, ε::Float64)
    # Total derivative dH/d(outer_j) = ∂H/∂θ · v_j + ∂H/∂x · dx*/d(outer_j).
    # Combined finite-difference perturbation captures both terms in one pair
    # of evaluations.
    lik_plus = _perturbed_autodiff_likelihood(lik, j, +ε)
    lik_minus = _perturbed_autodiff_likelihood(lik, j, -ε)
    x_plus = x_star .+ ε .* dx_j
    x_minus = x_star .- ε .* dx_j
    H_plus = _maybe_downcast_diagonal(GMRFs.loghessian(x_plus, lik_plus))
    H_minus = _maybe_downcast_diagonal(GMRFs.loghessian(x_minus, lik_minus))
    return _structurally_subtract(H_plus, H_minus, 2ε)
end

# DI.hessian returns a dense Matrix even when the underlying Hessian is
# structurally diagonal (the typical case for a sum-of-pointwise loglik).
# Sniff for off-diagonal zeros and downcast so the IFT path can preserve
# Q_prior's sparse pattern through `_assemble_q_post_dual`.
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

# (H_plus - H_minus) / (2ε), preserving Diagonal/sparse structure where
# applicable so downstream consumers can dispatch correctly.
_structurally_subtract(A::Diagonal, B::Diagonal, denom) = Diagonal((A.diag .- B.diag) ./ denom)
function _structurally_subtract(A::SparseMatrixCSC, B::SparseMatrixCSC, denom)
    A.colptr == B.colptr && A.rowval == B.rowval || return (A - B) / denom
    return SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), (A.nzval .- B.nzval) ./ denom)
end
_structurally_subtract(A::AbstractMatrix, B::AbstractMatrix, denom) = (A - B) ./ denom

# ----------------------------------------------------------------------------
# Q_post_dual builder — preserves Q_prior's exact sparse structure for
# diagonal Hessians (the dominant case via pointwise_loglik_func). For
# non-diagonal Hessians, falls back to algebraic subtraction (which can
# alter sparse structure).
# ----------------------------------------------------------------------------

function _assemble_q_post_dual(
        Q_prior::SparseMatrixCSC{Float64}, H_primal, H_dθ, OuterTag, ::Val{N}
    ) where {N}
    DualT = ForwardDiff.Dual{OuterTag, Float64, N}
    return _assemble_q_post_dual_impl(Q_prior, H_primal, H_dθ, DualT, Val(N))
end

# Diagonal H — write in place into a copy of Q_prior's nzval.
function _assemble_q_post_dual_impl(
        Q_prior::SparseMatrixCSC{Float64}, H_primal::Diagonal, H_dθ, ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    n = size(Q_prior, 1)
    PartialsT = ForwardDiff.Partials{N, Float64}
    zero_partials = PartialsT(ntuple(_ -> 0.0, Val(N)))
    nzval_dual = Vector{DualT}(undef, length(Q_prior.nzval))
    @inbounds for i in eachindex(Q_prior.nzval)
        nzval_dual[i] = DualT(Q_prior.nzval[i], zero_partials)
    end
    @inbounds for j in 1:n
        for k in nzrange(Q_prior, j)
            if Q_prior.rowval[k] == j
                # subtract H_primal[j,j] from primal, with partials = -H_dθ[k][j,j]
                primal = Q_prior.nzval[k] - H_primal.diag[j]
                partials = PartialsT(ntuple(d -> -H_dθ[d].diag[j], Val(N)))
                nzval_dual[k] = DualT(primal, partials)
                break
            end
        end
    end
    return SparseMatrixCSC(Q_prior.m, Q_prior.n, copy(Q_prior.colptr), copy(Q_prior.rowval), nzval_dual)
end

# Sparse non-diagonal H — match Q_prior's pattern, write H values where they
# overlap. Requires H's pattern to be a subset of Q_prior's; otherwise we'd
# silently drop H nonzeros outside Q_prior and return a wrong Q_post. Errors
# loudly in that case so the caller knows workspace reuse isn't applicable
# for this likelihood/prior combination.
function _assemble_q_post_dual_impl(
        Q_prior::SparseMatrixCSC{Float64}, H_primal::SparseMatrixCSC,
        H_dθ, ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    _check_h_pattern_subset(H_primal, Q_prior)
    n = size(Q_prior, 1)
    PartialsT = ForwardDiff.Partials{N, Float64}
    zero_partials = PartialsT(ntuple(_ -> 0.0, Val(N)))
    nzval_dual = Vector{DualT}(undef, length(Q_prior.nzval))
    @inbounds for i in eachindex(Q_prior.nzval)
        nzval_dual[i] = DualT(Q_prior.nzval[i], zero_partials)
    end
    # Look up H entries by (row, col) and subtract.
    @inbounds for col in 1:n
        for k in nzrange(Q_prior, col)
            row = Q_prior.rowval[k]
            h_primal = _sparse_lookup(H_primal, row, col)
            h_partials = PartialsT(ntuple(d -> -_sparse_lookup(H_dθ[d], row, col), Val(N)))
            primal = Q_prior.nzval[k] - h_primal
            nzval_dual[k] = DualT(primal, h_partials)
        end
    end
    return SparseMatrixCSC(Q_prior.m, Q_prior.n, copy(Q_prior.colptr), copy(Q_prior.rowval), nzval_dual)
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

# Generic / dense H — falls back to algebraic subtract; result loses sparse
# structure, which means workspace pattern checks will fail. Only used as a
# last resort for non-pointwise + non-sparse Hessians.
function _assemble_q_post_dual_impl(
        Q_prior::SparseMatrixCSC{Float64}, H_primal::AbstractMatrix,
        H_dθ, ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    n = size(Q_prior, 1)
    PartialsT = ForwardDiff.Partials{N, Float64}
    Q_dense_primal = Matrix(Q_prior) - H_primal
    Q_post_dual = Matrix{DualT}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        partials = PartialsT(ntuple(d -> -H_dθ[d][i, j], Val(N)))
        Q_post_dual[i, j] = DualT(Q_dense_primal[i, j], partials)
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
        ε::Float64 = sqrt(eps()),
        ga_kwargs...
    )
    OuterTag, N = _outer_tag_and_npartials(lik.hyperparams)
    DualT = ForwardDiff.Dual{OuterTag, Float64, N}

    # Step 1: primal Newton on stripped likelihood.
    primal_lik = _primal_autodiff_likelihood(lik)
    posterior_primal = GMRFs.gaussian_approximation(prior_gmrf, primal_lik; ga_kwargs...)
    x_star = GMRFs.mean(posterior_primal)
    n = length(x_star)

    # Step 2: per-partial ∂(∇_x loglik)/∂θ_j by central FD on θ. No nested AD —
    # outer θ-direction is FD, inner ∇_x is primal.
    grad_dθ = ntuple(j -> _grad_dθ_central(lik, x_star, j, ε), N)

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

    # Step 4: assemble Dual x*.
    x_star_dual = map(1:n) do i
        DualT(x_star[i], ForwardDiff.Partials{N, Float64}(ntuple(j -> dx[i, j], Val(N))))
    end

    # Step 5: assemble Dual Q_post via nzval-only update from primal Q_prior.
    # For the H tangent we need dH/d(outer_j) = ∂H/∂θ · v_j + ∂H/∂x · dx*/d(outer_j),
    # captured in one combined FD perturbation per direction.
    H_primal = _maybe_downcast_diagonal(GMRFs.loghessian(x_star, primal_lik))
    H_dθ = ntuple(j -> _hess_dθ_central(lik, x_star, view(dx, :, j), j, ε), N)
    Q_post_dual = _assemble_q_post_dual(prior_gmrf.precision, H_primal, H_dθ, OuterTag, Val(N))

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
