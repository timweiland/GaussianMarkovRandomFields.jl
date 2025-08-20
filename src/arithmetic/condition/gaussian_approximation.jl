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

    if obs_lik.indices === nothing
        # Non-indexed case: A = I (identity)
        A = 1.0 * I
    else
        # Indexed case: A selects the relevant components
        n_total = length(mean(prior_gmrf))
        n_obs = length(obs_lik.y)
        A = spzeros(n_obs, n_total)
        for (i, idx) in enumerate(obs_lik.indices)
            A[i, idx] = 1.0
        end
    end

    Q_ϵ = obs_lik.inv_σ²  # 1/σ² (scalar gets converted to scaled identity automatically)
    y = obs_lik.y
    b = zeros(length(y))  # No offset

    return linear_condition(prior_gmrf; A = A, Q_ϵ = Q_ϵ, y = y, b = b)
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

# MetaGMRF dispatches - preserve wrapper type and metadata
function gaussian_approximation(prior_mgmrf::MetaGMRF, obs_lik::ObservationLikelihood; kwargs...)
    posterior_gmrf = gaussian_approximation(prior_mgmrf.gmrf, obs_lik; kwargs...)
    return MetaGMRF(posterior_gmrf, prior_mgmrf.metadata)
end
