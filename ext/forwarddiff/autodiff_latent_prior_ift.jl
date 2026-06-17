# Forward-mode IFT for `gaussian_approximation` on an `AutoDiffLatentPrior` when the
# hyperparameters θ carry ForwardDiff.Dual partials (an outer hyperparameter-gradient
# pass, e.g. optimising the Laplace marginal likelihood over θ).
#
# Running the Newton loop with Dual θ is impossible — CHOLMOD can't factorize a Dual
# matrix, and the sparsity tracer can't coexist with Dual θ. So, exactly like the GMRF /
# AutoDiffLikelihood paths, we keep Newton in Float64 and get the θ-tangent analytically:
#
#   1. Strip θ to primal; primal Newton → x* (Float64) and the factored posterior.
#   2. ∂(neg_score)/∂θ at fixed x*: evaluate ∇ₓ logp_func(x*; θ_dual) (prior) plus
#      loggrad(x*, obs_lik) (lik, Dual iff obs carries Dual θ). The outer partials are
#      the partial θ-derivative of the score.
#   3. Solve Q_post · dx*/dθ_j = -∂(neg_score)/∂θ_j with the primal factorization.
#   4. Lift x* to Dual carrying dx*/dθ.
#   5. Dual Q_post = -∇²ₓ logp_func(x*_dual; θ_dual) - loghessian(x*_dual, obs_lik), the
#      prior Hessian computed with a *known-pattern* backend (ForwardDiff nests cleanly,
#      no re-tracing) — its Duals capture both ∂/∂θ and ∂/∂x·dx*/dθ.
#   6. Return a GMRF{Dual} (ConstrainedGMRF if the prior is constrained).
#
# The nested derivative passes use AutoForwardDiff explicitly (it nests under the outer
# Dual; the prior's own grad/hess backends may be reverse-mode and would not).

function GMRFs._nongaussian_dualhp_ift(
        prior::GMRFs.AutoDiffLatentPrior, obs_lik, θ_full::NamedTuple, ws; kwargs...
    )
    Tag, N = _outer_tag_and_npartials(θ_full)
    V = Float64
    DualT = ForwardDiff.Dual{Tag, V, N}
    PartialsT = ForwardDiff.Partials{N, V}

    # Step 1: primal Newton on stripped θ + stripped obs_lik.
    θ_primal = GMRFs._strip_ad_partials_hyperparams(θ_full)
    primal_obs = _primal_obs_lik(obs_lik)
    # Forward the workspace (#174): the primal Newton then reuses ws's symbolic factorization
    # (one symbolic factor across the whole θ-grid), and its converged Q_post factor backs the
    # IFT solves below. With ws === nothing this is the original fresh-cache path. Returns a
    # WorkspaceGMRF when ws is supplied, a (Constrained)GMRF otherwise.
    posterior_primal = GMRFs.gaussian_approximation(
        prior, primal_obs; θ = θ_primal, ws = ws, kwargs...
    )
    x_star = GMRFs.mean(posterior_primal)
    n = length(x_star)

    constraint_info = GMRFs.constraints(prior; θ_primal...)
    constraints_nt = constraint_info === nothing ? nothing :
        (A = constraint_info[1], e = constraint_info[2])

    # Step 2: ∂(neg_score)/∂θ at fixed x*.
    #   neg_score(x; θ) = -∇ₓ logp_func(x; θ) - loggrad(x, obs_lik)
    # Lift x* to Dual-with-zero-partials so the AD operator's buffers can hold the
    # θ-Dual-valued output; the outer partials of the result are ∂(neg_score)/∂θ.
    x_star_dual0 = DualT.(x_star)
    g_prior_dual = DI.gradient(z -> prior.logp_func(z; θ_full...), DI.AutoForwardDiff(), x_star_dual0)
    g_lik = GMRFs.loggrad(x_star, obs_lik)   # Float64 for a primal obs_lik; Dual if it carries θ
    neg_grad_dual = (-g_prior_dual) .- g_lik

    # Step 3: IFT solves Q_post · dx[:, j] = -∂(neg_score)/∂θ_j, reusing the primal
    # factorization. `alg` (for the Dual posterior in Step 6) comes from that same factor source.
    dx = Matrix{V}(undef, n, N)
    alg = if ws === nothing
        cache = GMRFs.linsolve_cache(GMRFs._base_gmrf(posterior_primal))
        b_saved = copy(cache.b)
        for j in 1:N
            for i in 1:n
                cache.b[i] = -ForwardDiff.partials(neg_grad_dual[i], j)
            end
            step = copy(solve!(cache).u)
            dx[:, j] .= GMRFs._constrain_step(step, cache, constraints_nt)
        end
        cache.b .= b_saved
        cache.alg
    else
        # Reuse the workspace's Q_post factorization. The primal build leaves it
        # unfactorized (by design — #167), so the first `workspace_solve` refactorizes once
        # and the remaining N-1 partials (plus the constraint Schur solves) reuse that factor.
        rhs = Vector{V}(undef, n)
        for j in 1:N
            for i in 1:n
                rhs[i] = -ForwardDiff.partials(neg_grad_dual[i], j)
            end
            step = GMRFs.workspace_solve(ws, rhs)
            dx[:, j] .= GMRFs._workspace_constrain_step(step, ws, constraints_nt)
        end
        # The workspace's CHOLMOD factor is Float64-only, so build the Dual posterior with the
        # default solver — which is what the no-ws path's `alg` resolves to anyway.
        nothing
    end

    # Step 4: Dual x* carrying dx*/dθ.
    x_star_dual = [
        DualT(x_star[i], PartialsT(ntuple(j -> dx[i, j], Val(N)))) for i in 1:n
    ]

    # Step 5: Dual Q_post. The prior Hessian uses a known-pattern backend (no tracer on
    # Duals); its Duals capture the total dQ/dθ (explicit θ + implicit x*(θ)).
    # Take the structural pattern from the prior's *own* primal local quadratic rather than
    # re-tracing `logp_func`: that reuses whatever Hessian backend the prior was built with,
    # so it works even when the density can't be traced (e.g. it solves an ODE). The
    # structural pattern is θ-independent, so detecting it at the primal point is exact.
    Q_prior_primal = GMRFs.local_quadratic(prior, x_star; θ_primal...).Q
    known_backend = GMRFs.known_pattern_hessian_backend(Q_prior_primal)
    H_prior_dual = DI.hessian(z -> prior.logp_func(z; θ_full...), known_backend, x_star_dual)
    H_lik_dual = GMRFs.loghessian(x_star_dual, obs_lik)
    Q_post_dual = (-H_prior_dual) - H_lik_dual

    # Step 6: result. `alg` was taken from the primal factor source in Step 3.
    base = GMRF(x_star_dual, Q_post_dual, alg)
    return constraints_nt === nothing ? base :
        GMRFs.ConstrainedGMRF(base, constraints_nt.A, constraints_nt.e)
end
