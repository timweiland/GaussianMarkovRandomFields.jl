# Forward-mode IFT support for `ConstrainedGMRF` priors. Includes the
# Dual-aware `_constraint_info` override (so the Q-path through
# `log_constraint_correction` is captured correctly) and dispatch hooks
# for Dual-prior and Float64-prior + Dual-obs cases.

# Override for Dual `base_gmrf`: ConstrainedGMRF's inner constructor stores
# `A_tilde_T` and `L_c` as Float64, so the default `_constraint_info` drops
# Q-path partials through `log_constraint_correction`. Here we rebuild
# `A_tilde_T` as a Dual matrix via implicit differentiation
# (Q * Ã^T = A'  ⇒  Q_v * Ã^T_p = -Q_p * Ã^T_v per partial direction),
# then form a Dual `L_c` by dense Cholesky of A * A_tilde_T_dual. The
# resulting constrained_mean and log_constraint_correction carry correct
# μ- and Q-path partials.
function GMRFs._constraint_info(
        base_gmrf::GMRFs.GMRF{D},
        A_dense::AbstractMatrix, e_vec::AbstractVector,
        A_tilde_T_v::Matrix{Float64},
        L_c_v::LinearAlgebra.Cholesky{Float64, Matrix{Float64}}
    ) where {D <: ForwardDiff.Dual}
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = length(base_gmrf)
    m = size(A_dense, 1)

    Q = GMRFs.precision_map(base_gmrf)

    # One Dual matvec per constraint row gives us all partials of
    # (Q * Ã^T_v)[:, i] at once: partial_k of (Q * Ã^T_v) = Q_p_k * Ã^T_v.
    QA_dual = Q * A_tilde_T_v   # n×m Dual matrix

    # Per-partial primal solve for the A_tilde_T tangents:
    # Q_v * Ã^T_p[:, i] = -Q_p_k * Ã^T_v[:, i]
    cache = GMRFs.linsolve_cache(base_gmrf)
    b_saved = copy(cache.b)
    A_tilde_T_partials = zeros(V, n, m, N)
    for k in 1:N, i in 1:m
        @inbounds for j in 1:n
            cache.b[j] = -ForwardDiff.partials(QA_dual[j, i], k)
        end
        A_tilde_T_partials[:, i, k] .= solve!(cache).u
    end
    cache.b .= b_saved

    # Reassemble A_tilde_T with Dual values.
    A_tilde_T_dual = Matrix{D}(undef, n, m)
    @inbounds for j in 1:n, i in 1:m
        A_tilde_T_dual[j, i] = ForwardDiff.Dual{Tag, V, N}(
            A_tilde_T_v[j, i],
            ForwardDiff.Partials{N, V}(ntuple(k -> A_tilde_T_partials[j, i, k], N)),
        )
    end

    # Dual L_c via dense m×m Cholesky.
    AAtt_dual = A_dense * A_tilde_T_dual
    L_c_dual = cholesky(Symmetric(AAtt_dual))

    μ_base = GMRFs.mean(base_gmrf)
    residual = A_dense * μ_base - e_vec
    resid_e = e_vec - A_dense * μ_base
    constrained_mean = μ_base - A_tilde_T_dual * (L_c_dual \ residual)
    log_constraint_correction =
        0.5 * (m * log(2π) + logdet(L_c_dual) + dot(resid_e, L_c_dual \ resid_e)) -
        0.5 * logdet(cholesky(Symmetric(A_dense * A_dense')))

    return constrained_mean, log_constraint_correction
end

function _primal_constrained_gmrf(prior::GMRFs.ConstrainedGMRF{<:ForwardDiff.Dual})
    primal_base = _primal_gmrf(prior.base_gmrf)
    return GMRFs.ConstrainedGMRF(primal_base, prior.constraint_matrix, prior.constraint_vector)
end

function _forwarddiff_gaussian_approximation_constrained(
        prior_gmrf::GMRFs.ConstrainedGMRF{D},
        obs_lik;
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    # --- Step 1: Primal forward pass ---
    primal_prior = _primal_constrained_gmrf(prior_gmrf)
    primal_obs_lik = _primal_obs_lik(obs_lik)
    posterior_primal = GMRFs.gaussian_approximation(primal_prior, primal_obs_lik; kwargs...)
    x_star = GMRFs.mean(posterior_primal)

    # --- Step 2: Compute ∂g/∂θ · θ̇ ---
    # Use the base GMRF (matching the forward pass which operates on the unconstrained GMRF)
    base_prior_dual = prior_gmrf.base_gmrf
    neg_grad_dual = GMRFs.∇ₓ_neg_log_posterior(base_prior_dual, obs_lik, x_star)

    # --- Step 3: Extract partials and solve N linear systems with constraint projection ---
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = length(x_star)

    cache = GMRFs.linsolve_cache(posterior_primal.base_gmrf)
    b_saved = copy(cache.b)
    constraints = GMRFs._extract_constraints(primal_prior)

    dx = Matrix{V}(undef, n, N)
    for j in 1:N
        for i in 1:n
            cache.b[i] = -ForwardDiff.partials(neg_grad_dual[i], j)
        end
        step = copy(solve!(cache).u)
        # Project onto constraint tangent space (KKT Schur complement)
        dx[:, j] .= GMRFs._constrain_step(step, cache, constraints)
    end
    cache.b .= b_saved

    # --- Step 4: Construct Dual-valued x* ---
    x_star_dual = map(1:n) do i
        ForwardDiff.Dual{Tag, V, N}(x_star[i], ForwardDiff.Partials{N, V}(ntuple(j -> dx[i, j], N)))
    end

    # --- Step 5: Compute posterior precision with Duals ---
    H_dual = GMRFs.loghessian(x_star_dual, obs_lik)
    Q_prior_dual = GMRFs.precision_matrix(base_prior_dual)
    Q_post_dual = Q_prior_dual - H_dual

    # --- Step 6: Construct result ConstrainedGMRF with Duals ---
    # Build the base GMRF with Dual values, then wrap in ConstrainedGMRF.
    # The ConstrainedGMRF constructor will compute correction and constrained_mean
    # using Dual arithmetic, so their derivatives are automatically tracked.
    alg = posterior_primal.base_gmrf.linsolve_cache.alg
    base_post_dual = GMRF(x_star_dual, Q_post_dual, alg)
    return GMRFs.ConstrainedGMRF(
        base_post_dual, prior_gmrf.constraint_matrix, prior_gmrf.constraint_vector
    )
end

# ConstrainedGMRF dispatch methods (Dual prior)
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.ConstrainedGMRF{D},
        obs_lik::GMRFs.ObservationLikelihood;
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    return _forwarddiff_gaussian_approximation_constrained(prior_gmrf, obs_lik; kwargs...)
end

function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.ConstrainedGMRF{D},
        obs_lik::GMRFs.NormalLikelihood{GMRFs.IdentityLink};
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    return _forwarddiff_gaussian_approximation_constrained(prior_gmrf, obs_lik; kwargs...)
end

function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.ConstrainedGMRF{D},
        obs_lik::GMRFs.LinearlyTransformedLikelihood{<:GMRFs.NormalLikelihood{GMRFs.IdentityLink}};
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    return _forwarddiff_gaussian_approximation_constrained(prior_gmrf, obs_lik; kwargs...)
end

# Float64 ConstrainedGMRF prior + Dual obs_lik routes through the generic
# obs-dual helper defined in `gmrf_gaussian_approximation.jl`.
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.ConstrainedGMRF{Float64}, obs_lik::_DualObsLik; kwargs...
    )
    return _forwarddiff_gaussian_approximation_obs_dual(prior_gmrf, obs_lik; kwargs...)
end

function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.ConstrainedGMRF{Float64},
        obs_lik::GMRFs.NormalLikelihood{GMRFs.IdentityLink, <:Any, <:ForwardDiff.Dual};
        kwargs...
    )
    return _forwarddiff_gaussian_approximation_obs_dual(prior_gmrf, obs_lik; kwargs...)
end
