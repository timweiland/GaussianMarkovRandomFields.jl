using LinearAlgebra
using SparseArrays
using NonlinearSolve
using LinearMaps

export gaussian_approximation

function neg_log_posterior(prior_gmrf::AbstractGMRF, obs_lik::ObservationLikelihood, x)
    return -logpdf(prior_gmrf, x) - loglik(x, obs_lik)
end

function ∇ₓ_neg_log_posterior(prior_gmrf::AbstractGMRF, obs_lik::ObservationLikelihood, x)
    return -gradlogpdf(prior_gmrf, x) - loggrad(x, obs_lik)
end

function ∇²ₓ_neg_log_posterior(prior_gmrf::AbstractGMRF, obs_lik::ObservationLikelihood, x)
    return precision_matrix(prior_gmrf) - loghessian(x, obs_lik)
end

"""
    gaussian_approximation(prior_gmrf, obs_lik) -> AbstractGMRF

Find Gaussian approximation to the posterior using Fisher scoring.

This function finds the mode of the posterior distribution and constructs a Gaussian
approximation around it using Fisher scoring (Newton-Raphson with Fisher information matrix).

# Arguments
- `prior_gmrf`: Prior GMRF distribution for the latent field
- `obs_lik`: Materialized observation likelihood (contains data and hyperparameters)

# Returns
- `posterior_gmrf::GMRF`: Gaussian approximation to the posterior p(x | θ, y)

# Example
```julia
# Set up components (done at higher level)
prior_gmrf = GMRF(μ_prior, Q_prior)
obs_model = ExponentialFamily(Poisson)
obs_lik = obs_model(y)  # Materialized once

# Find Gaussian approximation - returns a GMRF
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)
```
"""
function gaussian_approximation(prior_gmrf::GMRF, obs_lik::ObservationLikelihood; initial_guess = mean(prior_gmrf))
    nlf = NonlinearFunction{true}(
        (dg, u, p) -> (dg .= ∇ₓ_neg_log_posterior(p[1], p[2], u)),
        jac = (dh, u, p) -> (dh .= ∇²ₓ_neg_log_posterior(p[1], p[2], u)),
        jac_prototype = precision_matrix(prior_gmrf)
    )
    prob = NonlinearProblem(nlf, initial_guess, (prior_gmrf, obs_lik))
    sol = solve(
        prob,
        NewtonRaphson(linsolve = prior_gmrf.linsolve_cache.alg), abstol = 1.0e-6, reltol = 1.0e-6
    )
    x_star = sol.u
    return GMRF(x_star, ∇²ₓ_neg_log_posterior(prior_gmrf, obs_lik, x_star))
end

# Specialized dispatch for Normal observation likelihoods with identity link (conjugate prior case)
function gaussian_approximation(prior_gmrf::GMRF, obs_lik::NormalLikelihood{IdentityLink})
    # Normal observations with identity link: y ~ N(x, σ²I) - this is conjugate!
    # Equivalent to: y = A*x + 0 + ε, where ε ~ N(0, σ²I)

    Q_ϵ = obs_lik.inv_σ²  # 1/σ² (scalar gets converted to scaled identity automatically)
    y = obs_lik.y
    b = zeros(length(y))  # No offset

    if obs_lik.indices === nothing
        # Non-indexed case: A = I, so A' * Q_ϵ * A = Q_ϵ * I = Diagonal(fill(Q_ϵ, n))
        A = 1.0 * I
        n_total = length(mean(prior_gmrf))
        obs_precision_contrib = Diagonal(fill(obs_lik.inv_σ², n_total))
        return linear_condition(
            prior_gmrf; A = A, Q_ϵ = Q_ϵ, y = y, b = b,
            obs_precision_contrib = obs_precision_contrib
        )
    else
        # Indexed case: A' * Q_ϵ * A is diagonal with Q_ϵ at selected indices, 0 elsewhere
        n_total = length(mean(prior_gmrf))
        n_obs = length(obs_lik.y)
        A = spzeros(n_obs, n_total)
        diag_entries = zeros(n_total)
        @inbounds for (i, idx) in enumerate(obs_lik.indices)
            A[i, idx] = 1.0
            diag_entries[idx] = obs_lik.inv_σ²
        end
        return linear_condition(
            prior_gmrf; A = A, Q_ϵ = Q_ϵ, y = y, b = b,
            obs_precision_contrib = Diagonal(diag_entries)
        )
    end
end

# Specialized dispatch for linearly transformed Normal observation likelihoods (also conjugate)
function gaussian_approximation(prior_gmrf::GMRF, obs_lik::LinearlyTransformedLikelihood{<:NormalLikelihood{IdentityLink}})
    # Linearly transformed Normal with identity link: y ~ N(A*x, σ²I) - still conjugate!
    # This is exactly the linear conditioning setup: y = A*x + 0 + ε, where ε ~ N(0, σ²I)

    base_lik = obs_lik.base_likelihood
    A = obs_lik.design_matrix
    Q_ϵ = base_lik.inv_σ²  # 1/σ² (scalar gets converted to scaled identity automatically)
    y = base_lik.y
    b = zeros(length(y))  # No offset

    return linear_condition(prior_gmrf; A = A, Q_ϵ = Q_ϵ, y = y, b = b)
end

# Specialized dispatch for ConstrainedGMRF with Normal observations (conjugate case)
function gaussian_approximation(prior_constrained::ConstrainedGMRF, obs_lik::NormalLikelihood{IdentityLink})
    # Delegate to linear_condition for conjugate case
    if obs_lik.indices === nothing
        A = I
    else
        A = _build_index_matrix(obs_lik.indices, length(prior_constrained))
    end

    return linear_condition(
        prior_constrained;
        A = A,
        Q_ϵ = obs_lik.inv_σ²,
        y = obs_lik.y,
        b = zeros(length(obs_lik.y))
    )
end

# Helper function to build index matrix for indexed observations
function _build_index_matrix(indices, n_total)
    n_obs = length(indices)
    A = spzeros(n_obs, n_total)
    for (i, idx) in enumerate(indices)
        A[i, idx] = 1.0
    end
    return A
end

# Specialized dispatch for ConstrainedGMRF with linearly transformed Normal observations (conjugate case)
function gaussian_approximation(prior_constrained::ConstrainedGMRF, obs_lik::LinearlyTransformedLikelihood{<:NormalLikelihood{IdentityLink}})
    # Delegate to linear_condition for conjugate case
    base_lik = obs_lik.base_likelihood
    return linear_condition(
        prior_constrained;
        A = obs_lik.design_matrix,
        Q_ϵ = base_lik.inv_σ²,
        y = base_lik.y,
        b = zeros(length(base_lik.y))
    )
end

# ConstrainedGMRF dispatch - Newton method with constraint projection
"""
    gaussian_approximation(prior_constrained::ConstrainedGMRF, obs_lik::ObservationLikelihood; kwargs...)

Find Gaussian approximation to the constrained posterior using Newton optimization with constraint projection.

Alternates between Newton/Fisher scoring steps and projecting onto the constraint manifold,
which is mathematically equivalent to using Schur complements in the KKT approach for
constrained Newton optimization.

# Arguments
- `prior_constrained::ConstrainedGMRF`: Prior constrained GMRF distribution  
- `obs_lik::ObservationLikelihood`: Materialized observation likelihood

# Keyword Arguments
- `max_iter::Int=50`: Maximum number of Newton iterations
- `mean_change_tol::Real=1e-4`: Convergence tolerance for mean change
- `newton_dec_tol::Real=1e-5`: Newton decrement convergence tolerance  
- `verbose::Bool=false`: Print iteration information

# Returns
- `posterior_constrained::ConstrainedGMRF`: Constrained Gaussian approximation to posterior
"""
function gaussian_approximation(
        prior_constrained::ConstrainedGMRF,
        obs_lik::ObservationLikelihood;
        max_iter::Int = 50,
        mean_change_tol::Real = 1.0e-4,
        newton_dec_tol::Real = 1.0e-5,
        verbose::Bool = false
    )
    # Extract components from constrained prior
    base_gmrf = prior_constrained.base_gmrf
    A = prior_constrained.constraint_matrix
    e = prior_constrained.constraint_vector

    # Initialize with constrained prior mean
    x_k = mean(prior_constrained)

    cache = deepcopy(linsolve_cache(base_gmrf))
    Q_base = cache.A

    verbose && println("Starting Fisher scoring for ConstrainedGMRF...")

    for iter in 1:max_iter
        # Compute observation likelihood derivatives at current point
        H_k = loghessian(x_k, obs_lik)  # Hessian: ∇²ₓ log p(y|x)

        # Update precision: Q_new = Q_base - H_k (note: H_k contains negative of Hessian)
        Q_new = prepare_for_linsolve(Q_base - H_k, cache.alg)

        cache.A = Q_new
        neg_score_k = ∇ₓ_neg_log_posterior(base_gmrf, obs_lik, x_k)
        cache.b = neg_score_k
        step = solve!(cache).u
        μ_new = x_k - step
        newton_decrement = dot(neg_score_k, step)

        # Update cache matrix and create new GMRF with updated precision and information
        new_gmrf = GMRF(μ_new, Q_new; linsolve_cache = cache)

        # Wrap in ConstrainedGMRF with same constraints
        new_constrained = ConstrainedGMRF(new_gmrf, A, e)

        # Check convergence
        x_new = mean(new_constrained)

        mean_change = norm(x_new - x_k)
        mean_change_rel = mean_change / norm(x_k)

        verbose && println("  Iter $iter: Newton decrement = $(newton_decrement)")
        if (newton_decrement < newton_dec_tol) || (mean_change < mean_change_tol) || (mean_change_rel < mean_change_tol)
            verbose && println("  Converged after $iter iterations")
            return new_constrained
        end

        # Update for next iteration
        x_k = x_new
    end

    verbose && println("  Reached max_iter = $max_iter without convergence")

    # Return current best approximation
    H_k = loghessian(x_k, obs_lik)
    Q_final = prepare_for_linsolve(Q_base - H_k, cache.alg)
    cache.A = Q_final
    final_gmrf = GMRF(x_k, Q_final; linsolve_cache = cache)
    return ConstrainedGMRF(final_gmrf, A, e)
end

# MetaGMRF dispatches - preserve wrapper type and metadata
function gaussian_approximation(prior_mgmrf::MetaGMRF, obs_lik::ObservationLikelihood; kwargs...)
    posterior_gmrf = gaussian_approximation(prior_mgmrf.gmrf, obs_lik; kwargs...)
    return MetaGMRF(posterior_gmrf, prior_mgmrf.metadata)
end
