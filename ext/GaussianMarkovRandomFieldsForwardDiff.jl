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

# === WorkspaceGMRF ForwardDiff support ===
#
# When Dual numbers flow into WorkspaceGMRF construction, create the workspace
# from primal values (CHOLMOD can't handle Duals) while preserving Duals in
# the mean and precision fields for tangent propagation.

# update_precision! with a Dual-valued Q: strip to primal values before forwarding.
# This unblocks the LatentModel(ws; θ::Dual...) reuse path, which calls
# update_precision! with a Dual Q produced from Dual hyperparameters.
function GMRFs.update_precision!(
        ws::GMRFs.GMRFWorkspace, Q::SparseMatrixCSC{<:ForwardDiff.Dual}
    )
    Q_primal = SparseMatrixCSC(
        Q.m, Q.n, Q.colptr, Q.rowval, ForwardDiff.value.(Q.nzval)
    )
    return GMRFs.update_precision!(ws, Q_primal)
end

function _construct_forwarddiff_workspace_gmrf(
        mean::AbstractVector, Q::SparseMatrixCSC
    )
    T = promote_type(eltype(mean), eltype(Q))
    mean_T = eltype(mean) === T ? mean : convert(AbstractVector{T}, mean)
    Q_T = eltype(Q) === T ? Q : SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, convert(Vector{T}, Q.nzval))

    # Create workspace from primal values
    Q_primal = SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, ForwardDiff.value.(Q.nzval))
    ws = GMRFs.GMRFWorkspace(Q_primal)

    version = GMRFs._next_version!(ws)
    ws.loaded_version = version
    return GMRFs.WorkspaceGMRF{T, typeof(ws.backend), typeof(ws), Nothing}(
        Vector{T}(mean_T), copy(Q_T), ws, nothing, version
    )
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC
    )
    return _construct_forwarddiff_workspace_gmrf(mean, Q)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector,
        Q::SparseMatrixCSC{<:ForwardDiff.Dual}
    )
    return _construct_forwarddiff_workspace_gmrf(mean, Q)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC{<:ForwardDiff.Dual}
    )
    return _construct_forwarddiff_workspace_gmrf(mean, Q)
end

# 3-arg constructor with an existing workspace. The workspace stays primal;
# the WorkspaceGMRF holds Dual mean/precision for tangent propagation.
# Loading into the workspace happens lazily via the Dual `ensure_loaded!`
# override below (which strips to primal).
function _construct_forwarddiff_workspace_gmrf_with_ws(
        mean::AbstractVector, Q::SparseMatrixCSC, ws::GMRFs.GMRFWorkspace
    )
    T = promote_type(eltype(mean), eltype(Q))
    mean_T = eltype(mean) === T ? mean : convert(AbstractVector{T}, mean)
    Q_T = if eltype(Q) === T
        Q
    else
        SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, convert(Vector{T}, Q.nzval))
    end
    version = GMRFs._next_version!(ws)
    return GMRFs.WorkspaceGMRF{T, typeof(ws.backend), typeof(ws), Nothing}(
        Vector{T}(mean_T), copy(Q_T), ws, nothing, version
    )
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC,
        ws::GMRFs.GMRFWorkspace
    )
    return _construct_forwarddiff_workspace_gmrf_with_ws(mean, Q, ws)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector,
        Q::SparseMatrixCSC{<:ForwardDiff.Dual},
        ws::GMRFs.GMRFWorkspace
    )
    return _construct_forwarddiff_workspace_gmrf_with_ws(mean, Q, ws)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC{<:ForwardDiff.Dual},
        ws::GMRFs.GMRFWorkspace
    )
    return _construct_forwarddiff_workspace_gmrf_with_ws(mean, Q, ws)
end

# === WorkspaceGMRF ForwardDiff gaussian_approximation ===

function _primal_workspace_gmrf(prior::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual})
    μ_primal = ForwardDiff.value.(GMRFs.mean(prior))
    Q_primal = SparseMatrixCSC(
        prior.precision.m, prior.precision.n,
        prior.precision.colptr, prior.precision.rowval,
        ForwardDiff.value.(prior.precision.nzval)
    )
    return GMRFs.WorkspaceGMRF(μ_primal, Q_primal)
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

# Constrained Dual WorkspaceGMRF: workspace-reuse IFT gaussian_approximation
# with constraint projection. Mirrors _forwarddiff_gaussian_approximation_constrained
# but preserves the workspace across the Newton pass.
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


# Dual WorkspaceGMRFs hold Dual-valued precision but the workspace buffer is
# Float64. Extract primal values for reloading so version coherence works.
function GMRFs.ensure_loaded!(d::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual})
    ws = d.workspace
    if ws.loaded_version != d.version
        copyto!(ws.Q.nzval, ForwardDiff.value.(d.precision.nzval))
        GMRFs._invalidate!(ws)
        ws.loaded_version = d.version
    end
    return nothing
end

# logdetcov for Dual-valued WorkspaceGMRF: same approach as GMRF{Dual}
function logdetcov(x::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual})
    Qinv = GMRFs.selinv(x.workspace)
    primal = GMRFs.logdet_cov(x.workspace)
    # dot(Qinv, Q_dual) naturally produces a Dual via ForwardDiff overloads
    tangent = -dot(Qinv, x.precision)
    return ForwardDiff.Dual{ForwardDiff.tagtype(tangent)}(primal, ForwardDiff.partials(tangent)...)
end

# === Constrained WorkspaceGMRF with Duals ===
#
# The 5-arg constructor WorkspaceGMRF(μ, Q, ws, A, e) builds a ConstraintInfo
# which stores Ã^T = Q⁻¹A' and L_c = chol(A Ã^T) as Float64. For the
# unconstrained Dual flow, A_tilde_T_primal is enough because logpdf uses the
# unconstrained mean and Q. For the constrained flow, log_constraint_correction
# depends on Q through L_c and Ã^T — if we stored only primal values, the
# Q-path derivatives would be silently dropped.
#
# We resolve this by computing A_tilde_T and L_c with full Dual propagation
# (via implicit differentiation of Q Ã^T = A') and using those Dual values
# to form constrained_mean and log_constraint_correction. The struct-stored
# A_tilde_T / L_c stay primal — they're only used for sampling and var, which
# users don't typically differentiate through.

function _compute_constrained_duals(
        mean_T, Q::SparseMatrixCSC{<:ForwardDiff.Dual}, ws::GMRFs.GMRFWorkspace,
        A_dense::Matrix{Float64}, e_vec::Vector{Float64},
        A_tilde_T_v::Matrix{Float64}, log_AA_det::Float64
    )
    D = eltype(Q)
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = size(Q, 1)
    m = size(A_dense, 1)

    # Build Ã^T with Dual values via implicit diff.
    # Q Ã^T = A' (with A primal) gives, per partial direction k:
    #   Q_v Ã^T_p = -Q_p Ã^T_v
    A_tilde_T_partials = zeros(V, n, m, N)
    for k in 1:N
        Q_p_k_nzval = V[ForwardDiff.partials(Q.nzval[idx], k) for idx in eachindex(Q.nzval)]
        Q_p_k = SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, Q_p_k_nzval)
        for i in 1:m
            rhs = -(Q_p_k * @view(A_tilde_T_v[:, i]))
            A_tilde_T_partials[:, i, k] .= GMRFs.workspace_solve(ws, rhs)
        end
    end

    A_tilde_T_dual = Matrix{D}(undef, n, m)
    @inbounds for j in 1:n, i in 1:m
        A_tilde_T_dual[j, i] = ForwardDiff.Dual{Tag, V, N}(
            A_tilde_T_v[j, i],
            ForwardDiff.Partials{N, V}(ntuple(k -> A_tilde_T_partials[j, i, k], N)),
        )
    end

    # Dual L_c via dense Cholesky (m×m, small).
    AAtt_dual = A_dense * A_tilde_T_dual
    L_c_dual = cholesky(Symmetric(AAtt_dual))

    residual = A_dense * mean_T - e_vec
    resid_e = e_vec - A_dense * mean_T
    constrained_mean = mean_T - A_tilde_T_dual * (L_c_dual \ residual)
    log_constraint_correction =
        0.5 * (m * log(2π) + logdet(L_c_dual) + dot(resid_e, L_c_dual \ resid_e)) -
        0.5 * log_AA_det

    return constrained_mean, log_constraint_correction
end

function _construct_forwarddiff_constrained_workspace_gmrf(
        mean::AbstractVector, Q::SparseMatrixCSC, ws::GMRFs.GMRFWorkspace,
        A::AbstractMatrix, e::AbstractVector
    )
    T = promote_type(eltype(mean), eltype(Q))
    mean_T = convert(Vector{T}, mean)
    Q_T = if eltype(Q) === T
        Q
    else
        SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, convert(Vector{T}, Q.nzval))
    end

    # Load primal Q into ws so the primal factorization is current.
    Q_v_nzval = eltype(Q) <: ForwardDiff.Dual ?
        ForwardDiff.value.(Q.nzval) : Vector{Float64}(Q.nzval)
    Q_primal = SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, Q_v_nzval)
    GMRFs.update_precision!(ws, Q_primal)
    version = GMRFs._next_version!(ws)
    ws.loaded_version = version

    n = size(Q, 1)
    m = size(A, 1)
    A_dense = Matrix{Float64}(A)
    e_vec = Vector{Float64}(e)

    # Primal Ã^T = Q_v⁻¹ A' via m solves against the primal factorization.
    A_tilde_T_v = Matrix{Float64}(undef, n, m)
    for i in 1:m
        A_tilde_T_v[:, i] .= GMRFs.workspace_solve(ws, A_dense[i, :])
    end
    L_c_primal = cholesky(Symmetric(A_dense * A_tilde_T_v))
    log_AA_det = logdet(cholesky(Symmetric(A_dense * A_dense')))

    if eltype(Q) <: ForwardDiff.Dual
        constrained_mean, log_constraint_correction = _compute_constrained_duals(
            mean_T, Q, ws, A_dense, e_vec, A_tilde_T_v, log_AA_det
        )
    else
        # Dual-μ-only case: primal Ã^T / L_c are exact; μ-path Dual arithmetic
        # through the trailing `residual` terms is sufficient.
        residual = A_dense * mean_T - e_vec
        resid_e = e_vec - A_dense * mean_T
        constrained_mean = mean_T - A_tilde_T_v * (L_c_primal \ residual)
        log_constraint_correction =
            0.5 * (m * log(2π) + logdet(L_c_primal) + dot(resid_e, L_c_primal \ resid_e)) -
            0.5 * log_AA_det
    end

    ci = GMRFs.ConstraintInfo{T}(
        A_dense, e_vec, A_tilde_T_v, L_c_primal, constrained_mean, log_constraint_correction
    )
    B = typeof(ws.backend)
    return GMRFs.WorkspaceGMRF{T, B, typeof(ws), GMRFs.ConstraintInfo{T}}(
        mean_T, copy(Q_T), ws, ci, version
    )
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC,
        ws::GMRFs.GMRFWorkspace,
        A::AbstractMatrix,
        e::AbstractVector
    )
    return _construct_forwarddiff_constrained_workspace_gmrf(mean, Q, ws, A, e)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector,
        Q::SparseMatrixCSC{<:ForwardDiff.Dual},
        ws::GMRFs.GMRFWorkspace,
        A::AbstractMatrix,
        e::AbstractVector
    )
    return _construct_forwarddiff_constrained_workspace_gmrf(mean, Q, ws, A, e)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC{<:ForwardDiff.Dual},
        ws::GMRFs.GMRFWorkspace,
        A::AbstractMatrix,
        e::AbstractVector
    )
    return _construct_forwarddiff_constrained_workspace_gmrf(mean, Q, ws, A, e)
end

end
