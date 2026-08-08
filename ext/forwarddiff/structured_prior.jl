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
    # pass only consumes `mean` and the sparse snapshot.
    return GMRFs.StructuredPriorGMRF(μ_primal, prior.structure, Q_primal, prior.workspace, prior.cache)
end

function _forwarddiff_structured_ga(
        prior::GMRFs.StructuredPriorGMRF{D}, obs_lik; kwargs...
    ) where {D <: ForwardDiff.Dual}
    # Step 1: Primal forward pass.
    primal_prior = _primal_structured_prior(prior)
    primal_obs_lik = _primal_obs_lik(obs_lik)
    posterior_primal = GMRFs.gaussian_approximation(primal_prior, primal_obs_lik; kwargs...)
    x_star = GMRFs.mean(posterior_primal)

    # Step 2: ∇ₓ neg_log_posterior at x* with the Dual prior/likelihood.
    neg_grad_dual = prior.precision * (x_star .- prior.mean) .-
        GMRFs.loggrad(x_star, obs_lik)

    # Step 3: IFT tangent solves against the posterior workspace.
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = length(x_star)

    ws = posterior_primal.workspace
    dx = Matrix{V}(undef, n, N)
    for j in 1:N
        rhs_j = V[-ForwardDiff.partials(neg_grad_dual[i], j) for i in 1:n]
        dx[:, j] .= GMRFs.workspace_solve(ws, rhs_j)
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
    # Q_post factorization so the first consumer reuses it.
    return _own_workspace_factor!(ws, GMRFs.WorkspaceGMRF(x_star_dual, Q_post_sparse, ws))
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
    x_star = GMRFs.mean(posterior_primal)

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
        dx[:, j] .= GMRFs.workspace_solve(ws, rhs_j)
    end

    x_star_dual = map(1:n) do i
        ForwardDiff.Dual{Tag, V, N}(x_star[i], ForwardDiff.Partials{N, V}(ntuple(j -> dx[i, j], N)))
    end

    H_dual = GMRFs.loghessian(x_star_dual, obs_lik)
    Q_post_dual = prior.precision - H_dual
    Q_post_sparse = sparse(Q_post_dual)

    return _own_workspace_factor!(ws, GMRFs.WorkspaceGMRF(x_star_dual, Q_post_sparse, ws))
end

function GMRFs.gaussian_approximation(
        prior::GMRFs.StructuredPriorGMRF{Float64}, obs_lik::_DualObsLik; kwargs...
    )
    return _forwarddiff_structured_ga_obs_dual(prior, obs_lik; kwargs...)
end
