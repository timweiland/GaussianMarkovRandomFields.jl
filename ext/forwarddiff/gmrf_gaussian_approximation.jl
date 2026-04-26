# Forward-mode IFT for `gaussian_approximation` on plain `GMRF` priors,
# covering both the prior-Dual case and the Float64-prior + Dual-obs_lik case.

"""
    _primal_gmrf(prior::GMRF{<:ForwardDiff.Dual}) -> GMRF{Float64}

Extract a primal (non-Dual) GMRF from a Dual-valued one, preserving the solver algorithm.
"""
function _primal_gmrf(prior::GMRF{<:ForwardDiff.Dual})
    mu_primal = _primal_mean(GMRFs.mean(prior))
    Q_primal = _primal_precision(GMRFs.precision_matrix(prior))
    alg = GMRFs.linsolve_cache(prior).alg
    return GMRF(mu_primal, Q_primal, alg)
end

"""
    gaussian_approximation(prior_gmrf::GMRF{<:ForwardDiff.Dual}, obs_lik; kwargs...)

Forward-mode AD through `gaussian_approximation` using the Implicit Function Theorem.

Instead of propagating Dual numbers through the iterative Fisher scoring, this method:
1. Runs the primal forward pass to find the posterior mode x*
2. Uses the IFT to compute dx*/dθ via a linear solve with the already-factored Hessian
3. Computes the posterior precision tangent via loghessian evaluated at Dual x*
4. Returns a GMRF with Dual-valued mean and precision
"""
function _forwarddiff_gaussian_approximation(
        prior_gmrf::GMRF{D},
        obs_lik;
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    # --- Step 1: Primal forward pass ---
    primal_prior = _primal_gmrf(prior_gmrf)
    primal_obs_lik = _primal_obs_lik(obs_lik)
    posterior_primal = GMRFs.gaussian_approximation(primal_prior, primal_obs_lik; kwargs...)
    x_star = GMRFs.mean(posterior_primal)

    # --- Step 2: Compute ∂g/∂θ · θ̇ ---
    # Evaluate gradient with Dual prior but primal x* — the Dual part gives the JVP
    neg_grad_dual = GMRFs.∇ₓ_neg_log_posterior(prior_gmrf, obs_lik, x_star)

    # --- Step 3: Extract partials and solve N linear systems ---
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = length(x_star)

    # Reuse the factored posterior Hessian for back-substitution
    cache = GMRFs.linsolve_cache(posterior_primal)
    b_saved = copy(cache.b)

    # Solve H · ẋ*_j = -partials_j(neg_grad_dual) for each partial direction j
    dx = Matrix{V}(undef, n, N)
    for j in 1:N
        for i in 1:n
            cache.b[i] = -ForwardDiff.partials(neg_grad_dual[i], j)
        end
        dx[:, j] .= solve!(cache).u
    end
    cache.b .= b_saved

    # --- Step 4: Construct Dual-valued x* ---
    x_star_dual = map(1:n) do i
        ForwardDiff.Dual{Tag, V, N}(x_star[i], ForwardDiff.Partials{N, V}(ntuple(j -> dx[i, j], N)))
    end

    # --- Step 5: Compute posterior precision with Duals ---
    # Q_post = Q_prior - loghessian(x*, obs_lik)
    # Total derivative includes both explicit θ dependence and implicit x*(θ) dependence
    H_dual = GMRFs.loghessian(x_star_dual, obs_lik)
    Q_prior_dual = GMRFs.precision_matrix(prior_gmrf)
    Q_post_dual = Q_prior_dual - H_dual

    # --- Step 6: Construct result GMRF ---
    alg = GMRFs.linsolve_cache(posterior_primal).alg
    return GMRF(x_star_dual, Q_post_dual, alg)
end

# Forward-mode IFT when only obs_lik carries Dual hyperparameters and the prior
# is purely Float64. Used by both `GMRF{Float64}` and `ConstrainedGMRF{Float64}`
# priors via dispatch hooks defined in this file and in `constrained_gmrf.jl`.
function _forwarddiff_gaussian_approximation_obs_dual(
        prior_gmrf,
        obs_lik;
        kwargs...
    )
    D = _dual_type_from_obs_lik(obs_lik)

    # --- Step 1: Primal forward pass (prior is already Float64) ---
    primal_obs_lik = _primal_obs_lik(obs_lik)
    posterior_primal = GMRFs.gaussian_approximation(prior_gmrf, primal_obs_lik; kwargs...)
    x_star = GMRFs.mean(posterior_primal)

    # --- Step 2: Compute ∂g/∂θ · θ̇ ---
    neg_grad_dual = GMRFs.∇ₓ_neg_log_posterior(prior_gmrf, obs_lik, x_star)

    # --- Step 3: Extract partials and solve ---
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = length(x_star)

    cache = GMRFs.linsolve_cache(GMRFs._base_gmrf(posterior_primal))
    b_saved = copy(cache.b)

    dx = Matrix{V}(undef, n, N)
    for j in 1:N
        for i in 1:n
            cache.b[i] = -ForwardDiff.partials(neg_grad_dual[i], j)
        end
        step = copy(solve!(cache).u)
        dx[:, j] .= GMRFs._constrain_step(step, cache, GMRFs._extract_constraints(prior_gmrf))
    end
    cache.b .= b_saved

    # --- Step 4: Construct Dual-valued x* ---
    x_star_dual = map(1:n) do i
        ForwardDiff.Dual{Tag, V, N}(x_star[i], ForwardDiff.Partials{N, V}(ntuple(j -> dx[i, j], N)))
    end

    # --- Step 5: Compute posterior precision with Duals ---
    H_dual = GMRFs.loghessian(x_star_dual, obs_lik)
    Q_prior = GMRFs.precision_matrix(GMRFs._base_gmrf(prior_gmrf))
    Q_post_dual = Q_prior - H_dual

    # --- Step 6: Construct result ---
    alg = GMRFs.linsolve_cache(GMRFs._base_gmrf(posterior_primal)).alg
    base_post = GMRF(x_star_dual, Q_post_dual, alg)
    constraints = GMRFs._extract_constraints(prior_gmrf)
    return constraints === nothing ? base_post :
        GMRFs.ConstrainedGMRF(base_post, prior_gmrf.constraint_matrix, prior_gmrf.constraint_vector)
end

# Dispatch: Float64 prior + Dual obs_lik
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRF{Float64}, obs_lik::_DualObsLik; kwargs...
    )
    return _forwarddiff_gaussian_approximation_obs_dual(prior_gmrf, obs_lik; kwargs...)
end

# Disambiguation: conjugate Normal with Dual σ + Float64 prior
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRF{Float64},
        obs_lik::GMRFs.NormalLikelihood{GMRFs.IdentityLink, <:Any, <:ForwardDiff.Dual};
        kwargs...
    )
    return _forwarddiff_gaussian_approximation_obs_dual(prior_gmrf, obs_lik; kwargs...)
end

# Main dispatch: GMRF{Dual} with any ObservationLikelihood
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRF{D},
        obs_lik::GMRFs.ObservationLikelihood;
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    return _forwarddiff_gaussian_approximation(prior_gmrf, obs_lik; kwargs...)
end

# Disambiguation: conjugate Normal case (prevents ambiguity with specialized dispatch)
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRF{D},
        obs_lik::GMRFs.NormalLikelihood{GMRFs.IdentityLink};
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    return _forwarddiff_gaussian_approximation(prior_gmrf, obs_lik; kwargs...)
end

# Disambiguation: linearly transformed Normal case
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRF{D},
        obs_lik::GMRFs.LinearlyTransformedLikelihood{<:GMRFs.NormalLikelihood{GMRFs.IdentityLink}};
        kwargs...
    ) where {D <: ForwardDiff.Dual}
    return _forwarddiff_gaussian_approximation(prior_gmrf, obs_lik; kwargs...)
end
