"""
    GaussianMarkovRandomFieldsForwardDiff

"""
module GaussianMarkovRandomFieldsForwardDiff

import GaussianMarkovRandomFields as GMRFs
import GaussianMarkovRandomFields: GMRF
import Distributions: logdetcov

using ForwardDiff
using LinearAlgebra
using LinearMaps
using LinearSolve
using SparseArrays

const PrecisionLike = Union{LinearMaps.LinearMap, AbstractMatrix}

function _primal_mean(mean::AbstractVector)
    return ForwardDiff.value.(mean)
end

function _primal_precision(precision::AbstractMatrix)
    return ForwardDiff.value.(precision)
end

function _primal_precision(precision::SymTridiagonal)
    return SymTridiagonal(ForwardDiff.value.(precision.dv), ForwardDiff.value.(precision.ev))
end

function _primal_precision(precision::Diagonal)
    return Diagonal(ForwardDiff.value.(precision.diag))
end

function _primal_precision(precision::Symmetric{<:ForwardDiff.Dual, <:SparseMatrixCSC})
    return Symmetric(ForwardDiff.value.(precision.data))
end

function _primal_precision(precision::LinearMaps.LinearMap)
    return ForwardDiff.value.(to_matrix(precision))
end

# Build the LinearSolve cache with primal data because caches cannot handle Dual numbers.
function _forwarddiff_cache(mean::AbstractVector, precision::PrecisionLike, alg)
    configured_alg = GMRFs.configure_algorithm(alg)
    primal_rhs = copy(_primal_mean(mean))
    primal_precision = _primal_precision(precision)
    prepared_precision = GMRFs.prepare_for_linsolve(primal_precision, configured_alg)
    prob = LinearProblem(prepared_precision, primal_rhs)
    return init(prob, configured_alg)
end

# Reuse the standard constructor while injecting the Dual-safe LinearSolve cache.
function _construct_forwarddiff_gmrf(mean::AbstractVector, precision::PrecisionLike, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
    n = length(mean)
    n == size(precision, 1) == size(precision, 2) || throw(ArgumentError("size mismatch"))

    T = promote_type(eltype(mean), eltype(precision))
    mean_T = eltype(mean) === T ? mean : convert(AbstractVector{T}, mean)

    precision_T =
    if eltype(precision) === T
        precision
    elseif precision isa LinearMaps.LinearMap
        LinearMaps.LinearMap{T}(convert(AbstractMatrix{T}, to_matrix(precision)))
    else
        convert(AbstractMatrix{T}, precision)
    end

    cache = linsolve_cache === nothing ? _forwarddiff_cache(mean_T, precision_T, alg) : linsolve_cache

    return GMRFs.GMRF{T, typeof(mean_T), Nothing, typeof(precision_T), typeof(Q_sqrt), typeof(cache), typeof(rbmc_strategy)}(
        mean_T,
        nothing,
        precision_T,
        Q_sqrt,
        cache,
        rbmc_strategy,
    )
end

function GMRFs.GMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        precision::AbstractMatrix,
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

function GMRFs.GMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        precision::LinearMaps.LinearMap,
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

function GMRFs.GMRF(
        mean::AbstractVector,
        precision::AbstractMatrix{<:ForwardDiff.Dual},
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

function GMRFs.GMRF(
        mean::AbstractVector,
        precision::LinearMaps.LinearMap{<:ForwardDiff.Dual},
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

# NOTE: Combined Dual overloads remove ambiguities when both arguments carry Dual numbers.
function GMRFs.GMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        precision::AbstractMatrix{<:ForwardDiff.Dual},
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

function GMRFs.GMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        precision::LinearMaps.LinearMap{<:ForwardDiff.Dual},
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

function logdetcov(x::GMRF{<:ForwardDiff.Dual})
    Qinv = GMRFs.selinv(x.linsolve_cache)
    primal = GMRFs.logdet_cov(x.linsolve_cache)
    tangent = -dot(Qinv, x.precision)
    return ForwardDiff.Dual{ForwardDiff.tagtype(tangent)}(primal, ForwardDiff.partials(tangent)...)
end

# ============================================================================
# Forward-mode IFT for gaussian_approximation
# ============================================================================

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

# Extract primal values from observation likelihoods that may contain Dual hyperparameters.
# Default: identity (Poisson, Bernoulli, Binomial have no Real-typed hyperparameters)
_primal_obs_lik(obs_lik) = obs_lik

function _primal_obs_lik(lik::GMRFs.NormalLikelihood)
    return GMRFs.NormalLikelihood(
        lik.link, lik.y, ForwardDiff.value(lik.σ),
        ForwardDiff.value(lik.inv_σ²), ForwardDiff.value(lik.log_σ), lik.indices
    )
end

function _primal_obs_lik(lik::GMRFs.NegBinLikelihood)
    return GMRFs.NegBinLikelihood(
        lik.link, lik.y, ForwardDiff.value(lik.r), lik.indices, lik.logexposure
    )
end

function _primal_obs_lik(lik::GMRFs.GammaLikelihood)
    return GMRFs.GammaLikelihood(lik.link, lik.y, ForwardDiff.value(lik.phi), lik.indices)
end

function _primal_obs_lik(lik::GMRFs.StudentTLikelihood)
    return GMRFs.StudentTLikelihood(
        lik.link, lik.y, ForwardDiff.value(lik.σ), ForwardDiff.value(lik.ν),
        ForwardDiff.value(lik.w), ForwardDiff.value(lik.νp1),
        ForwardDiff.value(lik.σ_eff), lik.indices
    )
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

# ============================================================================
# Forward-mode IFT for ConstrainedGMRF
# ============================================================================

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

# ConstrainedGMRF dispatch methods
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

# ============================================================================
# Forward-mode IFT when only obs_lik carries Dual hyperparameters
# ============================================================================

# Detect Dual-valued observation likelihoods via their type parameter T
const _DualNormalLik = GMRFs.NormalLikelihood{<:GMRFs.LinkFunction, <:Any, <:ForwardDiff.Dual}
const _DualNegBinLik = GMRFs.NegBinLikelihood{<:GMRFs.LinkFunction, <:Any, <:Any, <:ForwardDiff.Dual}
const _DualGammaLik = GMRFs.GammaLikelihood{<:GMRFs.LinkFunction, <:Any, <:ForwardDiff.Dual}
const _DualStudentTLik = GMRFs.StudentTLikelihood{<:GMRFs.LinkFunction, <:Any, <:ForwardDiff.Dual}
const _DualObsLik = Union{_DualNormalLik, _DualNegBinLik, _DualGammaLik, _DualStudentTLik}

function _dual_type_from_obs_lik(::GMRFs.NormalLikelihood{L, I, D}) where {L, I, D}
    return D
end
function _dual_type_from_obs_lik(::GMRFs.NegBinLikelihood{L, I, O, D}) where {L, I, O, D}
    return D
end
function _dual_type_from_obs_lik(::GMRFs.GammaLikelihood{L, I, D}) where {L, I, D}
    return D
end
function _dual_type_from_obs_lik(::GMRFs.StudentTLikelihood{L, I, D}) where {L, I, D}
    return D
end

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

function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.ConstrainedGMRF{Float64}, obs_lik::_DualObsLik; kwargs...
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

function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.ConstrainedGMRF{Float64},
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

end
