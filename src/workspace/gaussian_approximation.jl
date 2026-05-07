using LinearAlgebra
using SparseArrays

"""
    _diagonal_indices(Q::SparseMatrixCSC) -> Vector{Int}

Return the nzval indices corresponding to the diagonal entries of Q.
"""
function _diagonal_indices(Q::SparseMatrixCSC)
    n = size(Q, 1)
    indices = Vector{Int}(undef, n)
    rows = rowvals(Q)
    for col in 1:n
        for idx in nzrange(Q, col)
            if rows[idx] == col
                indices[col] = idx
                break
            end
        end
    end
    return indices
end

"""
    _sparse_hessian_map(Q::SparseMatrixCSC, H::SparseMatrixCSC) -> Vector{Int}

Build an index map from H's nonzero positions to Q's nzval positions.
Returns a vector `map` where `Q.nzval[map[k]]` corresponds to `H.nzval[k]`.
Errors if H has nonzeros outside Q's sparsity pattern.
"""
function _sparse_hessian_map(Q::SparseMatrixCSC, H::SparseMatrixCSC)
    Q_rows = rowvals(Q)
    H_rows = rowvals(H)
    hmap = Vector{Int}(undef, nnz(H))

    k = 0
    for col in 1:size(H, 2)
        q_range = nzrange(Q, col)
        q_ptr = first(q_range)
        q_end = last(q_range)
        for h_idx in nzrange(H, col)
            h_row = H_rows[h_idx]
            k += 1
            while q_ptr <= q_end && Q_rows[q_ptr] < h_row
                q_ptr += 1
            end
            if q_ptr <= q_end && Q_rows[q_ptr] == h_row
                hmap[k] = q_ptr
            else
                error(
                    "Hessian has nonzero at ($h_row, $col) which is outside the workspace Q sparsity pattern."
                )
            end
        end
    end
    return hmap
end

function _subtract_diagonal_hessian!(
        ws::GMRFWorkspace, H::Diagonal, diag_indices::Vector{Int}
    )
    h_diag = H.diag
    @inbounds for i in eachindex(h_diag)
        ws.Q.nzval[diag_indices[i]] -= h_diag[i]
    end
    return nothing
end

function _subtract_sparse_hessian!(
        ws::GMRFWorkspace, H::SparseMatrixCSC, hess_map::Vector{Int}
    )
    h_nzval = nonzeros(H)
    @inbounds for k in eachindex(h_nzval)
        ws.Q.nzval[hess_map[k]] -= h_nzval[k]
    end
    return nothing
end

"""
    _update_hessian!(ws, H_k, prior_nzval, diag_idx, sparse_hess_map)

Restore prior values into workspace Q and subtract the Hessian in-place.
Returns the (possibly initialized) sparse_hess_map.
"""
function _update_hessian!(ws, H_k, prior_nzval, diag_idx, sparse_hess_map)
    copyto!(ws.Q.nzval, prior_nzval)
    if H_k isa Diagonal
        _subtract_diagonal_hessian!(ws, H_k, diag_idx)
    elseif H_k isa SparseMatrixCSC
        if sparse_hess_map === nothing
            sparse_hess_map = _sparse_hessian_map(ws.Q, H_k)
        end
        _subtract_sparse_hessian!(ws, H_k, sparse_hess_map)
    else
        H_sparse = sparse(H_k)
        if sparse_hess_map === nothing
            sparse_hess_map = _sparse_hessian_map(ws.Q, H_sparse)
        end
        _subtract_sparse_hessian!(ws, H_sparse, sparse_hess_map)
    end
    _invalidate!(ws)
    # ws.Q no longer matches any WorkspaceGMRF's snapshot — it's a transient
    # Newton iterate (Q_prior - H_k). Reset the ownership tag so a subsequent
    # `logpdf` / `var` / ... on any WorkspaceGMRF sharing this workspace
    # triggers `ensure_loaded!` to reload the owner's snapshot. Without this,
    # the quadratic-form term uses the correct precision (from
    # `d.precision`) but the log-determinant comes from the workspace's
    # transient Q, silently mis-pairing them.
    ws.loaded_version = 0
    return sparse_hess_map
end

"""
    _workspace_constrain_step(step, ws, constraints)

Project Newton step onto constraint tangent space via KKT Schur complement,
using the workspace's factorization for the m sparse solves.
"""
_workspace_constrain_step(step, ws, ::Nothing) = step
_workspace_constrain_step(step, ws::GMRFWorkspace, constraints::ConstraintInfo) =
    _workspace_constrain_with_matrix(step, ws, constraints.matrix)
_workspace_constrain_step(step, ws::GMRFWorkspace, constraints::NamedTuple) =
    _workspace_constrain_with_matrix(step, ws, constraints.A)

function _workspace_constrain_with_matrix(step, ws::GMRFWorkspace, A)
    m = size(A, 1)
    n = length(step)
    A_tilde_T = Matrix{eltype(step)}(undef, n, m)
    for i in 1:m
        A_tilde_T[:, i] .= workspace_solve(ws, A[i, :])
    end
    L_c = cholesky(Symmetric(A * A_tilde_T))
    return step - A_tilde_T * (L_c \ (A * step))
end

"""
    _snapshot_Q(ws::GMRFWorkspace) -> SparseMatrixCSC

Create a snapshot of the workspace's current Q values.
Shares colptr/rowval (immutable), copies nzval.
"""
function _snapshot_Q(ws::GMRFWorkspace{T}) where {T}
    return SparseMatrixCSC(
        ws.Q.m, ws.Q.n, ws.Q.colptr, ws.Q.rowval, copy(ws.Q.nzval)
    )
end

"""
    _build_result(ws, x, prior_constraints) -> WorkspaceGMRF

Build the result WorkspaceGMRF from the converged workspace state.
Re-applies constraints if the prior had them.
"""
function _build_result(ws::GMRFWorkspace, x::Vector, prior_constraints::Nothing)
    Q_post = _snapshot_Q(ws)
    return WorkspaceGMRF(x, Q_post, ws)
end

function _build_result(ws::GMRFWorkspace, x::Vector, prior_constraints::ConstraintInfo)
    Q_post = _snapshot_Q(ws)
    return WorkspaceGMRF(x, Q_post, ws, prior_constraints.matrix, prior_constraints.vector)
end

function _build_result(ws::GMRFWorkspace, x::Vector, prior_constraints::NamedTuple)
    Q_post = _snapshot_Q(ws)
    return WorkspaceGMRF(x, Q_post, ws, prior_constraints.A, prior_constraints.e)
end

"""
    gaussian_approximation(prior::WorkspaceGMRF, obs_lik::ObservationLikelihood; kwargs...)

Workspace-aware Gaussian approximation via Fisher scoring. Uses the
workspace's factorisation engine for numeric-only refactorisation on
each Newton step.
"""
function gaussian_approximation(
        prior::WorkspaceGMRF,
        obs_lik::ObservationLikelihood;
        x0::Union{Nothing, AbstractVector} = nothing,
        max_iter::Int = 50,
        mean_change_tol::Real = 1.0e-4,
        newton_dec_tol::Real = 1.0e-5,
        adaptive_stepsize::Bool = true,
        max_linesearch_iter::Int = 10,
        verbose::Bool = false
    )
    ws = prior.workspace
    # Make sure ws.Q reflects this prior's snapshot before the loop reads
    # the prior's `(Q, h)` out of it on iter 1.
    ensure_loaded!(prior)
    x_init = x0 === nothing ? copy(mean(prior)) : copy(x0)
    return _workspace_newton_loop(
        prior, ws, obs_lik, prior.constraints, x_init;
        max_iter, mean_change_tol, newton_dec_tol,
        adaptive_stepsize, max_linesearch_iter, verbose,
    )
end

"""
    _workspace_newton_loop(prior, ws, obs_lik, constraints, x_init; ...) -> WorkspaceGMRF

Shared workspace-backed Newton loop. The prior side is queried via
`prior_quadratic(prior, x_k)` per iterate; for materialised
`WorkspaceGMRF` priors `(Q, h)` are constant in `x_k`, while for the
`LatentPrior` adapter they re-evaluate via `local_quadratic`. The
workspace's symbolic factor is reused as long as the sparsity pattern is
constant, which is true whenever the prior's structural couplings are
fixed by the model graph (the realistic case).
"""
function _workspace_newton_loop(
        prior, ws::GMRFWorkspace, obs_lik::ObservationLikelihood,
        constraints, x_init::AbstractVector;
        max_iter::Int,
        mean_change_tol::Real,
        newton_dec_tol::Real,
        adaptive_stepsize::Bool,
        max_linesearch_iter::Int,
        verbose::Bool,
    )
    diag_idx = _diagonal_indices(ws.Q)
    sparse_hess_map = nothing
    x_k = copy(x_init)
    α = 1.0

    verbose && println("Starting workspace Fisher scoring...")

    for iter in 1:max_iter
        lq = prior_quadratic(prior, x_k)
        H_k = loghessian(x_k, obs_lik)
        g_l = loggrad(x_k, obs_lik)

        sparse_hess_map = _update_hessian!(ws, H_k, lq.Q.nzval, diag_idx, sparse_hess_map)
        ensure_numeric!(ws)

        neg_score_k = (lq.Q * x_k - lq.h) .- g_l
        step = workspace_solve(ws, neg_score_k)
        step = _workspace_constrain_step(step, ws, constraints)

        if adaptive_stepsize
            obj_current = -lq.logp_ref - loglik(x_k, obs_lik)
            step_accepted = false
            local x_candidate
            for ls_iter in 1:max_linesearch_iter
                x_candidate = x_k - α * step
                logp_cand = prior_quadratic(prior, x_candidate).logp_ref
                obj_candidate = -logp_cand - loglik(x_candidate, obs_lik)
                if obj_candidate <= obj_current
                    α = sqrt(α)
                    step_accepted = true
                    verbose && ls_iter > 1 &&
                        println("    Accepted at α=$(round(α^2, digits = 3)) after $ls_iter backtracks")
                    break
                else
                    α *= 0.1
                    verbose && println("    Backtrack: α=$(round(α, digits = 4))")
                    if α * norm(step, Inf) < newton_dec_tol / 1000
                        step_accepted = true
                        break
                    end
                end
            end
            x_new = step_accepted ? x_candidate : (x_k - α * step)
        else
            x_new = x_k - step
        end

        newton_decrement = dot(neg_score_k, step)
        mean_change = norm(x_new - x_k)
        mean_change_rel = mean_change / max(norm(x_k), 1.0e-10)
        verbose && println(
            "  Iter $iter: Newton dec = $(round(newton_decrement, sigdigits = 3)), α = $(round(α, digits = 3))"
        )

        if (newton_decrement < newton_dec_tol) ||
                (mean_change < mean_change_tol) ||
                (mean_change_rel < mean_change_tol)
            verbose && println("  Converged after $iter iterations")
            return _workspace_build_result(prior, ws, obs_lik, constraints, x_new, diag_idx, sparse_hess_map)
        end

        x_k = x_new
    end

    verbose && println("  Reached max_iter without convergence")
    return _workspace_build_result(prior, ws, obs_lik, constraints, x_k, diag_idx, sparse_hess_map)
end

# Refresh ws.Q to Q_post(x_final) so the snapshot used by `_build_result`
# reflects the converged iterate (the loop's last `_update_hessian!` may
# have happened at `x_k`, not `x_new`; line-search reload of Q_prior via
# `ensure_loaded!` could also have left ws.Q out of sync).
function _workspace_build_result(prior, ws, obs_lik, constraints, x_final, diag_idx, sparse_hess_map)
    lq_final = prior_quadratic(prior, x_final)
    H_final = loghessian(x_final, obs_lik)
    _update_hessian!(ws, H_final, lq_final.Q.nzval, diag_idx, sparse_hess_map)
    ensure_numeric!(ws)
    return _build_result(ws, x_final, constraints)
end

# Conjugate Normal case: fall back to the existing linear_condition path
function gaussian_approximation(
        prior::WorkspaceGMRF,
        obs_lik::NormalLikelihood{IdentityLink};
        x0::Union{Nothing, AbstractVector} = nothing,
        max_iter::Int = 50,
        mean_change_tol::Real = 1.0e-4,
        newton_dec_tol::Real = 1.0e-5,
        adaptive_stepsize::Bool = true,
        max_linesearch_iter::Int = 10,
        verbose::Bool = false
    )
    if has_constraints(prior)
        ci = prior.constraints
        temp_base = GMRF(prior.mean, prior.precision)
        temp_constrained = ConstrainedGMRF(temp_base, ci.matrix, ci.vector)
        result = gaussian_approximation(temp_constrained, obs_lik)
        Q_post = sparse(precision_matrix(result))
        update_precision!(prior.workspace, Q_post)
        return WorkspaceGMRF(
            mean(_base_gmrf(result)), Q_post, prior.workspace, ci.matrix, ci.vector
        )
    else
        temp_gmrf = GMRF(prior.mean, prior.precision)
        result = gaussian_approximation(temp_gmrf, obs_lik)
        Q_post = sparse(precision_matrix(result))
        update_precision!(prior.workspace, Q_post)
        return WorkspaceGMRF(mean(result), Q_post, prior.workspace)
    end
end
