using LinearAlgebra
using SparseArrays
using LinearMaps
using CliqueTrees.Multifrontal: ChordalCholesky

export gaussian_approximation

# Sparse-preserving subtraction for Hermitian matrices
hermdiff(A::Hermitian, B) = Hermitian(parent(A) - B, Symbol(A.uplo))

function neg_log_posterior(prior_gmrf::AbstractGMRF, obs_lik::ObservationLikelihood, x)
    return -logpdf(prior_gmrf, x) - loglik(x, obs_lik)
end

function ∇ₓ_neg_log_posterior(prior_gmrf::AbstractGMRF, obs_lik::ObservationLikelihood, x)
    return -gradlogpdf(prior_gmrf, x) - loggrad(x, obs_lik)
end

function ∇²ₓ_neg_log_posterior(prior_gmrf::AbstractGMRF, obs_lik::ObservationLikelihood, x)
    return precision_matrix(prior_gmrf) - loghessian(x, obs_lik)
end

# Private dispatch: extract constraints if present
_extract_constraints(::GMRF) = nothing
_extract_constraints(constrained::ConstrainedGMRF) = (A = constrained.constraint_matrix, e = constrained.constraint_vector)
_extract_constraints(::ChordalGMRF) = nothing

# Private dispatch: apply constraints to result
_apply_constraints(gmrf::GMRF, ::Nothing) = gmrf
_apply_constraints(gmrf::GMRF, constraints::NamedTuple) = ConstrainedGMRF(gmrf, constraints.A, constraints.e)

# Private dispatch: extract base GMRF for optimization
_base_gmrf(gmrf::GMRF) = gmrf
_base_gmrf(constrained::ConstrainedGMRF) = constrained.base_gmrf
_base_gmrf(gmrf::ChordalGMRF) = gmrf

# Compute the constrained Newton step via the KKT Schur complement.
# Solves H⁻¹Aᵀ using the existing linsolve cache (m sparse solves, m = #constraints),
# then removes the constraint-normal component so AΔx = 0.
_constrain_step(step, cache, ::Nothing) = step
function _constrain_step(step, cache, constraints::NamedTuple)
    A = constraints.A
    m = size(A, 1)
    n = length(step)

    # Solve H · X = Aᵀ column-by-column (reuses current cache factorization)
    A_tilde_T = Matrix{eltype(step)}(undef, n, m)
    saved_b = copy(cache.b)
    for i in 1:m
        cache.b .= @view(A[i, :])
        A_tilde_T[:, i] .= solve!(cache).u
    end
    cache.b .= saved_b

    # Schur complement: remove constraint-normal component
    L_c = cholesky(Symmetric(A * A_tilde_T))
    return step - A_tilde_T * (L_c \ (A * step))
end

# Set the matrix in the linsolve cache to Q
_update_linsolve_cache!(cache, Q) = _update_linsolve_cache_inner!(cache, Q, cache.alg)

function _update_linsolve_cache_inner!(cache, Q, alg)
    return cache.A = Q
end

function _update_linsolve_cache_inner!(cache, Q, alg::LinearSolve.DefaultLinearSolver)
    actual_alg = LinearSolve.algchoice_to_alg(Symbol(alg.alg))
    return _update_linsolve_cache_inner!(cache, Q, actual_alg)
end

function _update_linsolve_cache_inner!(cache, Q, alg::LinearSolve.LDLtFactorization)
    # LDLtFactorization mutates the original matrix, so we need to copy it
    # Causes super nasty and sneaky bugs if we don't do this!
    return cache.A = copy(Q)
end

# Solver abstraction for gaussian_approximation Newton iteration.
# Allows shared iteration logic for both LinearSolve-backed GMRF and ChordalCholesky-backed ChordalGMRF.
_ga_init_solver(gmrf::GMRF) = deepcopy(linsolve_cache(gmrf))
_ga_init_solver(gmrf::ChordalGMRF) = copy(gmrf.F)

function _ga_update_and_solve!(solver, Q_base, H_k, b, ::GMRF)
    Q_new = prepare_for_linsolve(Q_base - H_k, solver.alg)
    _update_linsolve_cache!(solver, Q_new)
    solver.b = b
    return Q_new, copy(solve!(solver).u)
end

function _ga_update_and_solve!(solver, Q_base, H_k, b, ::ChordalGMRF)
    Q_new = hermdiff(Q_base, H_k)
    copyto!(solver, Q_new)
    cholesky!(solver)
    return Q_new, solver \ b
end

function _ga_make_posterior(x, Q, solver, prior::Union{GMRF, ConstrainedGMRF}, constraints)
    new_gmrf = GMRF(x, Q; linsolve_cache = solver)
    return _apply_constraints(new_gmrf, constraints)
end

function _ga_make_posterior(x, Q, solver, prior::ChordalGMRF, ::Nothing)
    return ChordalGMRF(x, Q, solver)
end

"""
    gaussian_approximation(prior_gmrf, obs_lik; kwargs...) -> AbstractGMRF

Find Gaussian approximation to the posterior using Fisher scoring.

This function finds the mode of the posterior distribution and constructs a Gaussian
approximation around it using Fisher scoring (Newton-Raphson with Fisher information matrix).

Works for `GMRF`, `ConstrainedGMRF`, and `ChordalGMRF` priors, automatically handling
constraint projection when needed.

# Arguments
- `prior_gmrf`: Prior GMRF distribution for the latent field (GMRF, ConstrainedGMRF, or ChordalGMRF)
- `obs_lik`: Materialized observation likelihood (contains data and hyperparameters)

# Keyword Arguments
- `max_iter::Int=50`: Maximum number of Fisher scoring iterations
- `mean_change_tol::Real=1e-4`: Convergence tolerance for mean change
- `newton_dec_tol::Real=1e-5`: Newton decrement convergence tolerance
- `adaptive_stepsize::Bool=true`: Enable adaptive stepsize with backtracking line search
- `max_linesearch_iter::Int=10`: Maximum line search iterations per Newton step
- `verbose::Bool=false`: Print iteration information

# Returns
- Gaussian approximation to the posterior p(x | θ, y) (same type as input prior)

# Example
```julia
# Set up components
prior_gmrf = GMRF(μ_prior, Q_prior)
obs_model = ExponentialFamily(Poisson)
obs_lik = obs_model(y)

# Find Gaussian approximation - uses adaptive stepsize by default
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)

# For well-conditioned problems, disable adaptive stepsize for speed
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik; adaptive_stepsize=false)
```
"""
function gaussian_approximation(
        prior_gmrf::Union{GMRF, ConstrainedGMRF, ChordalGMRF},
        obs_lik::ObservationLikelihood;
        x0::Union{Nothing, AbstractVector} = nothing,
        max_iter::Int = 50,
        mean_change_tol::Real = 1.0e-4,
        newton_dec_tol::Real = 1.0e-5,
        adaptive_stepsize::Bool = true,
        max_linesearch_iter::Int = 10,
        verbose::Bool = false
    )
    # Extract base GMRF and constraints (nothing for GMRF/ChordalGMRF)
    base_gmrf = _base_gmrf(prior_gmrf)
    constraints = _extract_constraints(prior_gmrf)

    # Initialize with provided starting point or prior mean
    x_k = x0 === nothing ? copy(mean(prior_gmrf)) : copy(x0)

    solver = _ga_init_solver(base_gmrf)
    Q_base = precision_matrix(base_gmrf)

    # Adaptive stepsize state (persists across outer iterations)
    α = 1.0

    verbose && println("Starting Fisher scoring...")

    for iter in 1:max_iter
        H_k = loghessian(x_k, obs_lik)
        neg_score_k = ∇ₓ_neg_log_posterior(base_gmrf, obs_lik, x_k)
        Q_new, step = _ga_update_and_solve!(solver, Q_base, H_k, neg_score_k, base_gmrf)

        # For constrained problems, project step onto constraint tangent space
        # via the KKT Schur complement. No-op when constraints are nothing.
        step = _constrain_step(step, solver, constraints)

        # Apply step with adaptive line search or full step
        if adaptive_stepsize
            obj_current = neg_log_posterior(base_gmrf, obs_lik, x_k)
            accept = false

            for ls_iter in 1:max_linesearch_iter
                candidate = x_k - α * step
                obj_candidate = neg_log_posterior(base_gmrf, obs_lik, candidate)

                if obj_candidate <= obj_current
                    x_new = candidate
                    α = sqrt(α)
                    accept = true
                    verbose && ls_iter > 1 && println("    Accepted at α=$(round(α^2, digits = 3)) after $ls_iter backtracks")
                    break
                else
                    α *= 0.1
                    verbose && println("    Backtrack: α=$(round(α, digits = 4))")

                    if α * norm(step, Inf) < newton_dec_tol / 1000
                        x_new = candidate
                        accept = true
                        break
                    end
                end
            end

            if !accept
                x_new = x_k - α * step
            end
        else
            x_new = x_k - step
        end

        newton_decrement = dot(neg_score_k, step)
        mean_change = norm(x_new - x_k)
        mean_change_rel = mean_change / max(norm(x_k), 1.0e-10)

        verbose && println("  Iter $iter: Newton dec = $(round(newton_decrement, sigdigits = 3)), α = $(round(α, digits = 3))")
        if (newton_decrement < newton_dec_tol) || (mean_change < mean_change_tol) || (mean_change_rel < mean_change_tol)
            verbose && println("  Converged after $iter iterations")
            return _ga_make_posterior(x_new, Q_new, solver, prior_gmrf, constraints)
        end

        x_k = x_new
    end

    verbose && println("  Reached max_iter = $max_iter without convergence")

    # Return current best approximation at final x_k
    H_k = loghessian(x_k, obs_lik)
    neg_score_k = ∇ₓ_neg_log_posterior(base_gmrf, obs_lik, x_k)
    Q_final, _ = _ga_update_and_solve!(solver, Q_base, H_k, neg_score_k, base_gmrf)
    return _ga_make_posterior(x_k, Q_final, solver, prior_gmrf, constraints)
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

# MetaGMRF dispatches - preserve wrapper type and metadata
function gaussian_approximation(prior_mgmrf::MetaGMRF, obs_lik::ObservationLikelihood; kwargs...)
    posterior_gmrf = gaussian_approximation(prior_mgmrf.gmrf, obs_lik; kwargs...)
    return MetaGMRF(posterior_gmrf, prior_mgmrf.metadata)
end
