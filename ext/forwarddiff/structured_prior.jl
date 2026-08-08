# ForwardDiff support for structured priors (StructuredPriorGMRF).
#
# The structured prior needs no Dual constructor shims: unlike a WorkspaceGMRF
# prior, it never writes into the (Float64) joint workspace, so Dual-valued
# means/precisions flow through construction unchanged. What the extension
# provides is:
#
#   1. `_primal_float` — lets a first evaluation with Dual hyperparameters
#      seed the Float64 per-factor engines.
#   2. Dual sparse-leaf log-determinants — primal logdet from the factor
#      engine plus a selected-inverse tangent, `d logdet(Q) = tr(Q⁻¹ dQ)`,
#      contracted at *factor* scale via `selinv_dot` (never a joint selinv).
#   3. IFT `gaussian_approximation` methods mirroring the WorkspaceGMRF ones:
#      primal Newton pass, tangent solves against the posterior factorization.

GMRFs._primal_float(x::ForwardDiff.Dual) = GMRFs._primal_float(ForwardDiff.value(x))

# ----------------------------------------------------------------------------
# Dual sparse-leaf log-determinants
# ----------------------------------------------------------------------------

# Cached leaf: factor engine holds the primal factorization; the tangent is
# the selected inverse contracted against the Dual factor values.
function GMRFs._sparse_leaf_logdet(
        engine::GMRFs.GMRFWorkspace, Q::SparseMatrixCSC{<:ForwardDiff.Dual}
    )
    GMRFs._same_pattern(Q, engine.Q) ||
        throw(ArgumentError("factor pattern changed since the structured prior cache was built."))
    GMRFs._leaf_sync!(engine, ForwardDiff.value.(Q.nzval))
    primal = logdet(engine)
    tangent = GMRFs.selinv_dot(engine, Q)
    return ForwardDiff.Dual{ForwardDiff.tagtype(tangent)}(primal, ForwardDiff.partials(tangent)...)
end

# Cacheless leaf (one-shot paths like `prior_logdensity`): ephemeral engine.
function GMRFs._sparse_leaf_logdet(Q::SparseMatrixCSC{<:ForwardDiff.Dual})
    Q_primal = SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, ForwardDiff.value.(Q.nzval))
    engine = GMRFs.GMRFWorkspace(Q_primal)
    primal = logdet(engine)
    tangent = GMRFs.selinv_dot(engine, Q)
    return ForwardDiff.Dual{ForwardDiff.tagtype(tangent)}(primal, ForwardDiff.partials(tangent)...)
end

# ----------------------------------------------------------------------------
# Dual constraint gram: S = A_i Q_i⁻¹ A_iᵀ with tangent dS = -Vᵀ dQ_i V
# ----------------------------------------------------------------------------
#
# `Vᵀ Q_dual V` has primal value S (since Q V = A_iᵀ at the primal point) and
# partials `Vᵀ dQ V`, so the Dual gram is assembled by pairing the primal
# values with the *negated* partials — the derivative of the inverse.

function GMRFs._constraint_gram_sparse(
        engine::GMRFs.GMRFWorkspace, Q::SparseMatrixCSC{<:ForwardDiff.Dual},
        A_i::SparseMatrixCSC
    )
    GMRFs._same_pattern(Q, engine.Q) ||
        throw(ArgumentError("factor pattern changed since the structured prior cache was built."))
    GMRFs._leaf_sync!(engine, ForwardDiff.value.(Q.nzval))
    m_i, n_i = size(A_i)
    V = Matrix{Float64}(undef, n_i, m_i)
    for r in 1:m_i
        V[:, r] = GMRFs.workspace_solve(engine, Vector(A_i[r, :]))
    end
    S_primal = Matrix(A_i * V)
    T = V' * (Q * V)
    S = map(S_primal, T) do sp, t
        ForwardDiff.Dual{ForwardDiff.tagtype(t)}(sp, (-ForwardDiff.partials(t))...)
    end
    return S, V
end

# ----------------------------------------------------------------------------
# Dual StructuredPriorGMRF — IFT gaussian_approximation
# ----------------------------------------------------------------------------
#
# Mirrors `_forwarddiff_workspace_ga`: primal forward pass, then tangent
# solves against the joint workspace (factorized at Q_post by the primal
# Newton loop). The prior side of the IFT gradient is a plain matvec with the
# Dual snapshot — no prior-side factorization anywhere.

function _primal_structured_prior(prior::GMRFs.StructuredPriorGMRF{<:ForwardDiff.Dual})
    μ_primal = ForwardDiff.value.(prior.mean)
    Q = prior.precision
    Q_primal = SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, ForwardDiff.value.(Q.nzval))
    # The Dual-valued lazy structure rides along unused: the primal Newton
    # pass only consumes `mean`, the sparse snapshot, and the constraint
    # system (A, e), all of which are primal.
    return GMRFs.StructuredPriorGMRF(
        μ_primal, prior.structure, Q_primal, prior.workspace, prior.cache,
        prior.constraints
    )
end

function _forwarddiff_structured_ga(
        prior::GMRFs.StructuredPriorGMRF{D}, obs_lik; kwargs...
    ) where {D <: ForwardDiff.Dual}
    # Step 1: Primal forward pass (constraint projection included when the
    # prior is constrained).
    primal_prior = _primal_structured_prior(prior)
    primal_obs_lik = _primal_obs_lik(obs_lik)
    posterior_primal = GMRFs.gaussian_approximation(primal_prior, primal_obs_lik; kwargs...)
    ci = posterior_primal.constraints
    # Constrained: use the Newton iterate (unconstrained mean field, satisfies
    # the constraints by projection), mirroring the WorkspaceGMRF variant.
    x_star = ci === nothing ? GMRFs.mean(posterior_primal) : posterior_primal.mean

    # Step 2: ∇ₓ neg_log_posterior at x* with the Dual prior/likelihood.
    neg_grad_dual = prior.precision * (x_star .- prior.mean) .-
        GMRFs.loggrad(x_star, obs_lik)

    # Step 3: IFT tangent solves against the posterior workspace (KKT
    # projection onto the constraint tangent space when constrained).
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = length(x_star)

    ws = posterior_primal.workspace
    dx = Matrix{V}(undef, n, N)
    for j in 1:N
        rhs_j = V[-ForwardDiff.partials(neg_grad_dual[i], j) for i in 1:n]
        step = GMRFs.workspace_solve(ws, rhs_j)
        if ci !== nothing
            step = step - ci.A_tilde_T * (ci.L_c \ (ci.matrix * step))
        end
        dx[:, j] .= step
    end

    # Step 4: Dual x*.
    x_star_dual = map(1:n) do i
        ForwardDiff.Dual{Tag, V, N}(x_star[i], ForwardDiff.Partials{N, V}(ntuple(j -> dx[i, j], N)))
    end

    # Step 5: Dual posterior precision on the workspace pattern.
    H_dual = GMRFs.loghessian(x_star_dual, obs_lik)
    Q_post_dual = prior.precision - H_dual
    Q_post_sparse = sparse(Q_post_dual)

    # Step 6: Dual posterior WorkspaceGMRF; tag it as the owner of the live
    # Q_post factorization so the first consumer reuses it. Constrained
    # posteriors lift the primal Ã^T / L_c from the primal pass (already
    # computed at Q_post) instead of redoing the m constraint solves.
    if ci === nothing
        return _own_workspace_factor!(ws, GMRFs.WorkspaceGMRF(x_star_dual, Q_post_sparse, ws))
    else
        log_AA_det = logdet(cholesky(Symmetric(ci.matrix * ci.matrix')))
        return _own_workspace_factor!(
            ws,
            _build_constrained_dual_workspace_gmrf(
                x_star_dual, Q_post_sparse, ws,
                ci.matrix, ci.vector, ci.A_tilde_T, ci.L_c,
                log_AA_det, posterior_primal.version
            ),
        )
    end
end

function GMRFs.gaussian_approximation(
        prior::GMRFs.StructuredPriorGMRF{<:ForwardDiff.Dual},
        obs_lik::GMRFs.ObservationLikelihood;
        kwargs...
    )
    return _forwarddiff_structured_ga(prior, obs_lik; kwargs...)
end

# ----------------------------------------------------------------------------
# Float64 StructuredPriorGMRF + Dual likelihood
# ----------------------------------------------------------------------------

function _forwarddiff_structured_ga_obs_dual(
        prior::GMRFs.StructuredPriorGMRF{Float64}, obs_lik; kwargs...
    )
    D = _dual_type_from_obs_lik(obs_lik)

    primal_obs_lik = _primal_obs_lik(obs_lik)
    posterior_primal = GMRFs.gaussian_approximation(prior, primal_obs_lik; kwargs...)
    ci = posterior_primal.constraints
    x_star = ci === nothing ? GMRFs.mean(posterior_primal) : posterior_primal.mean

    # Only the likelihood term carries Duals here.
    neg_grad_dual = prior.precision * (x_star .- prior.mean) .-
        GMRFs.loggrad(x_star, obs_lik)

    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = length(x_star)

    ws = posterior_primal.workspace
    dx = Matrix{V}(undef, n, N)
    for j in 1:N
        rhs_j = V[-ForwardDiff.partials(neg_grad_dual[i], j) for i in 1:n]
        step = GMRFs.workspace_solve(ws, rhs_j)
        if ci !== nothing
            step = step - ci.A_tilde_T * (ci.L_c \ (ci.matrix * step))
        end
        dx[:, j] .= step
    end

    x_star_dual = map(1:n) do i
        ForwardDiff.Dual{Tag, V, N}(x_star[i], ForwardDiff.Partials{N, V}(ntuple(j -> dx[i, j], N)))
    end

    H_dual = GMRFs.loghessian(x_star_dual, obs_lik)
    Q_post_dual = prior.precision - H_dual
    Q_post_sparse = sparse(Q_post_dual)

    if ci === nothing
        return _own_workspace_factor!(ws, GMRFs.WorkspaceGMRF(x_star_dual, Q_post_sparse, ws))
    else
        log_AA_det = logdet(cholesky(Symmetric(ci.matrix * ci.matrix')))
        return _own_workspace_factor!(
            ws,
            _build_constrained_dual_workspace_gmrf(
                x_star_dual, Q_post_sparse, ws,
                ci.matrix, ci.vector, ci.A_tilde_T, ci.L_c,
                log_AA_det, posterior_primal.version
            ),
        )
    end
end

function GMRFs.gaussian_approximation(
        prior::GMRFs.StructuredPriorGMRF{Float64}, obs_lik::_DualObsLik; kwargs...
    )
    return _forwarddiff_structured_ga_obs_dual(prior, obs_lik; kwargs...)
end
