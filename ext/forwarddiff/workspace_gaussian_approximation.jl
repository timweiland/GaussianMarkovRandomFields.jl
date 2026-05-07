# Forward-mode IFT for `gaussian_approximation` on `WorkspaceGMRF` priors:
#   - Float64 prior + Dual obs_lik (via _forwarddiff_workspace_ga_obs_dual)
#   - Dual prior, unconstrained (via _forwarddiff_workspace_ga)
#   - Dual prior, constrained (via _forwarddiff_workspace_ga_constrained)

# ----------------------------------------------------------------------------
# Float64 WorkspaceGMRF + Dual obs_lik
# ----------------------------------------------------------------------------
#
# The primal Newton iteration runs via the primal `gaussian_approximation`
# dispatch (Float64 prior, Float64 obs_lik). After convergence, we evaluate
# ∇ₓ neg_log_posterior with the Dual obs_lik to extract obs-hyperparameter
# tangents, IFT-solve them against the posterior workspace factorization,
# and reconstruct a Dual-valued WorkspaceGMRF posterior.

function _forwarddiff_workspace_ga_obs_dual(
        prior_gmrf::GMRFs.WorkspaceGMRF{Float64},
        obs_lik;
        kwargs...
    )
    D = _dual_type_from_obs_lik(obs_lik)

    # Step 1: Primal forward pass with Float64 obs_lik via the primal path.
    primal_obs_lik = _primal_obs_lik(obs_lik)
    posterior_primal = GMRFs.gaussian_approximation(prior_gmrf, primal_obs_lik; kwargs...)
    # `mean(::WorkspaceGMRF)` returns the constrained mean when constraints are
    # present (the unconstrained `posterior_primal.mean` field is NOT the
    # constrained KKT optimum for the conjugate Normal path, where it's set to
    # the unprojected linear-condition mean).
    x_star = GMRFs.mean(posterior_primal)

    # Step 2: ∇ₓ neg_log_posterior(x*) with Dual obs_lik. The prior is
    # Float64 so the first term contributes no Duals; the loggrad term
    # carries the obs-hyperparameter partials.
    neg_grad_dual = prior_gmrf.precision * (x_star .- prior_gmrf.mean) .-
        GMRFs.loggrad(x_star, obs_lik)

    # Step 3: IFT tangent solves against the posterior workspace
    # (factorized at Q_post by the primal forward pass).
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = length(x_star)

    ws = posterior_primal.workspace
    constrained = posterior_primal.constraints !== nothing
    ci = constrained ? posterior_primal.constraints : nothing

    dx = Matrix{V}(undef, n, N)
    for j in 1:N
        rhs_j = V[-ForwardDiff.partials(neg_grad_dual[i], j) for i in 1:n]
        step = GMRFs.workspace_solve(ws, rhs_j)
        if constrained
            A = ci.matrix
            step = step - ci.A_tilde_T * (ci.L_c \ (A * step))
        end
        dx[:, j] .= step
    end

    # Step 4: Construct Dual x*.
    x_star_dual = map(1:n) do i
        ForwardDiff.Dual{Tag, V, N}(x_star[i], ForwardDiff.Partials{N, V}(ntuple(j -> dx[i, j], N)))
    end

    # Step 5: Posterior precision with Duals (Q_prior is Float64; H_dual
    # carries both x*-tangent and obs-hyperparameter partials).
    H_dual = GMRFs.loghessian(x_star_dual, obs_lik)
    Q_post_dual = prior_gmrf.precision - H_dual
    Q_post_sparse = sparse(Q_post_dual)

    # Step 6: Build result, preserving constraints if present. For the
    # constrained branch, lift the primal Ã^T / L_c from posterior_primal's
    # ConstraintInfo (already computed at Q_post during the primal pass) to
    # avoid a second factorization + m constraint solves.
    if prior_gmrf.constraints === nothing
        return GMRFs.WorkspaceGMRF(x_star_dual, Q_post_sparse, ws)
    else
        ci_post = posterior_primal.constraints
        A_dense = ci_post.matrix
        log_AA_det = logdet(cholesky(Symmetric(A_dense * A_dense')))
        return _build_constrained_dual_workspace_gmrf(
            x_star_dual, Q_post_sparse, ws,
            A_dense, ci_post.vector, ci_post.A_tilde_T, ci_post.L_c,
            log_AA_det, posterior_primal.version
        )
    end
end

function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.WorkspaceGMRF{Float64}, obs_lik::_DualObsLik; kwargs...
    )
    return _forwarddiff_workspace_ga_obs_dual(prior_gmrf, obs_lik; kwargs...)
end

# Disambiguation: Float64 workspace prior + conjugate Normal with Dual σ.
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.WorkspaceGMRF{Float64},
        obs_lik::GMRFs.NormalLikelihood{GMRFs.IdentityLink, <:Any, <:ForwardDiff.Dual};
        kwargs...
    )
    return _forwarddiff_workspace_ga_obs_dual(prior_gmrf, obs_lik; kwargs...)
end

# ----------------------------------------------------------------------------
# Dual WorkspaceGMRF prior — unconstrained
# ----------------------------------------------------------------------------

function _primal_workspace_gmrf(prior::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual})
    μ_primal = ForwardDiff.value.(prior.mean)
    Q_primal = SparseMatrixCSC(
        prior.precision.m, prior.precision.n,
        prior.precision.colptr, prior.precision.rowval,
        ForwardDiff.value.(prior.precision.nzval)
    )
    # Reuse the prior's existing workspace rather than allocating and
    # symbolically factorizing a fresh one. This matches the constrained
    # variant (`_primal_constrained_workspace_gmrf`) and is the whole
    # point of the workspace abstraction.
    return GMRFs.WorkspaceGMRF(μ_primal, Q_primal, prior.workspace)
end

function _forwarddiff_workspace_ga(
        prior_gmrf::GMRFs.WorkspaceGMRF{D},
        obs_lik;
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    # Step 1: Primal forward pass
    primal_prior = _primal_workspace_gmrf(prior_gmrf)
    primal_obs_lik = _primal_obs_lik(obs_lik)
    posterior_primal = GMRFs.gaussian_approximation(primal_prior, primal_obs_lik; kwargs...)
    x_star = GMRFs.mean(posterior_primal)

    # Step 2: Evaluate gradient with Dual prior at primal x*
    neg_grad_dual = GMRFs.∇ₓ_neg_log_posterior(prior_gmrf, obs_lik, x_star)

    # Step 3: Solve IFT linear systems using the posterior workspace
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = length(x_star)

    ws = posterior_primal.workspace
    dx = Matrix{V}(undef, n, N)
    for j in 1:N
        rhs_j = [-ForwardDiff.partials(neg_grad_dual[i], j) for i in 1:n]
        dx[:, j] .= GMRFs.workspace_solve(ws, rhs_j)
    end

    # Step 4: Construct Dual-valued x*
    x_star_dual = map(1:n) do i
        ForwardDiff.Dual{Tag, V, N}(x_star[i], ForwardDiff.Partials{N, V}(ntuple(j -> dx[i, j], N)))
    end

    # Step 5: Compute posterior precision with Duals
    H_dual = GMRFs.loghessian(x_star_dual, obs_lik)
    Q_prior_dual = GMRFs.precision_matrix(prior_gmrf)
    Q_post_dual = Q_prior_dual - H_dual

    # Step 6: Return WorkspaceGMRF with Dual values and primal workspace
    Q_post_sparse = sparse(Q_post_dual)
    return GMRFs.WorkspaceGMRF(x_star_dual, Q_post_sparse, ws)
end

# Tried splitting via `WorkspaceGMRF{D, B, W, Nothing}` and
# `WorkspaceGMRF{D, B, W, C} where {C<:ConstraintInfo}` dispatches, but the
# constrained variant didn't fire reliably alongside the primal package's
# `gaussian_approximation(::WorkspaceGMRF, ...)` method (Julia method
# resolution preferred the primal definition for the constrained Dual
# case). Falling back to a runtime check keeps things unambiguous.
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.WorkspaceGMRF{D},
        obs_lik::GMRFs.ObservationLikelihood;
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    if prior_gmrf.constraints === nothing
        return _forwarddiff_workspace_ga(prior_gmrf, obs_lik; kwargs...)
    else
        return _forwarddiff_workspace_ga_constrained(prior_gmrf, obs_lik; kwargs...)
    end
end

# Disambiguation: Dual WorkspaceGMRF + conjugate Normal
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.WorkspaceGMRF{D},
        obs_lik::GMRFs.NormalLikelihood{GMRFs.IdentityLink};
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    if prior_gmrf.constraints === nothing
        return _forwarddiff_workspace_ga(prior_gmrf, obs_lik; kwargs...)
    else
        return _forwarddiff_workspace_ga_constrained(prior_gmrf, obs_lik; kwargs...)
    end
end

# ----------------------------------------------------------------------------
# Dual WorkspaceGMRF prior — constrained
# ----------------------------------------------------------------------------
# Workspace-reuse IFT gaussian_approximation with constraint projection.
# Mirrors `_forwarddiff_gaussian_approximation_constrained` but preserves
# the workspace across the Newton pass.

function _primal_constrained_workspace_gmrf(
        prior::GMRFs.WorkspaceGMRF{D}
    ) where {D <: ForwardDiff.Dual}
    μ_primal = ForwardDiff.value.(prior.mean)
    Q_primal = SparseMatrixCSC(
        prior.precision.m, prior.precision.n,
        prior.precision.colptr, prior.precision.rowval,
        ForwardDiff.value.(prior.precision.nzval)
    )
    ci = prior.constraints
    return GMRFs.WorkspaceGMRF(μ_primal, Q_primal, prior.workspace, ci.matrix, ci.vector)
end

function _forwarddiff_workspace_ga_constrained(
        prior_gmrf::GMRFs.WorkspaceGMRF{D},
        obs_lik;
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    # Step 1: Primal forward pass with constraints.
    primal_prior = _primal_constrained_workspace_gmrf(prior_gmrf)
    primal_obs_lik = _primal_obs_lik(obs_lik)
    posterior_primal = GMRFs.gaussian_approximation(primal_prior, primal_obs_lik; kwargs...)
    # Newton iterate (unconstrained mean field, satisfies constraints by projection).
    x_star = posterior_primal.mean

    # Step 2: Evaluate ∇ neg_log_posterior with the Dual prior at primal x*.
    # Inline computation avoids bumping ws.loaded_version (which would cause
    # ensure_loaded! to replace Q_post in ws with Q_prior, breaking IFT solves).
    # ∇ₓ neg_log_posterior(x) = Q_prior (x - μ_prior) - loggrad(x, obs_lik)
    # (unconstrained base gradient; constraint projection applied to IFT step below)
    neg_grad_dual = prior_gmrf.precision * (x_star .- prior_gmrf.mean) .-
        GMRFs.loggrad(x_star, obs_lik)

    # Step 3: IFT tangent solves with KKT constraint projection.
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = length(x_star)

    ws = posterior_primal.workspace
    ci_primal = posterior_primal.constraints
    A = ci_primal.matrix

    dx = Matrix{V}(undef, n, N)
    for j in 1:N
        rhs_j = V[-ForwardDiff.partials(neg_grad_dual[i], j) for i in 1:n]
        step = GMRFs.workspace_solve(ws, rhs_j)
        # Project onto constraint tangent space: step - Ã^T (L_c \ (A step))
        step_proj = step - ci_primal.A_tilde_T * (ci_primal.L_c \ (A * step))
        dx[:, j] .= step_proj
    end

    # Step 4: Construct Dual x*.
    x_star_dual = map(1:n) do i
        ForwardDiff.Dual{Tag, V, N}(x_star[i], ForwardDiff.Partials{N, V}(ntuple(j -> dx[i, j], N)))
    end

    # Step 5: Posterior precision with Duals.
    H_dual = GMRFs.loghessian(x_star_dual, obs_lik)
    Q_prior_dual = GMRFs.precision_matrix(prior_gmrf)
    Q_post_dual = Q_prior_dual - H_dual
    Q_post_sparse = sparse(Q_post_dual)

    # Step 6: Build Dual constrained WorkspaceGMRF with the prior's constraints.
    ci_prior = prior_gmrf.constraints
    return GMRFs.WorkspaceGMRF(
        x_star_dual, Q_post_sparse, ws, ci_prior.matrix, ci_prior.vector
    )
end
