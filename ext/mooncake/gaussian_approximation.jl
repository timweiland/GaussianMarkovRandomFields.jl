# `gaussian_approximation` for every supported prior type: a tangent-free
# primitive for the Newton solve, then one differentiable Newton step at the
# converged mode (the Implicit Function Theorem correction).

const MooncakeGAPrior = Union{ChordalGMRF, GMRF, WorkspaceGMRF, ConstrainedGMRF}

function gaussian_approximation_notangent(prior::MooncakeGAPrior, obslik::ObservationLikelihood; kwargs...)
    return gaussian_approximation(prior, obslik; kwargs...)
end

@is_primitive MinimalCtx Tuple{typeof(gaussian_approximation_notangent), MooncakeGAPrior, ObservationLikelihood}
@is_primitive MinimalCtx Tuple{typeof(Core.kwcall), Any, typeof(gaussian_approximation_notangent), MooncakeGAPrior, ObservationLikelihood}

function Mooncake.rrule!!(
        ::CoDual{typeof(gaussian_approximation_notangent)},
        cdprior::CoDual{<:MooncakeGAPrior},
        cdobslik::CoDual{<:ObservationLikelihood},
    )
    prior = primal(cdprior)
    obslik = primal(cdobslik)
    posterior = gaussian_approximation_notangent(prior, obslik)

    # Accept any rdata: GMRF posteriors carry non-trivial (but irrelevant)
    # rdata from the RNG state inside the default RBMC strategy.
    function pullback!!(::Any)
        return NoRData(), Mooncake.zero_rdata(prior), Mooncake.zero_rdata(obslik)
    end

    return CoDual(posterior, fdata(zero_tangent(posterior))), pullback!!
end

function Mooncake.rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        cdkwargs::CoDual,
        ::CoDual{typeof(gaussian_approximation_notangent)},
        cdprior::CoDual{<:MooncakeGAPrior},
        cdobslik::CoDual{<:ObservationLikelihood},
    )
    prior = primal(cdprior)
    obslik = primal(cdobslik)
    kwargs = primal(cdkwargs)
    posterior = gaussian_approximation_notangent(prior, obslik; kwargs...)

    function pullback!!(::Any)
        return NoRData(), NoRData(), NoRData(), Mooncake.zero_rdata(prior), Mooncake.zero_rdata(obslik)
    end

    return CoDual(posterior, fdata(zero_tangent(posterior))), pullback!!
end

# --- The IFT correction ---
# One differentiable Newton step at the converged, tangent-free mode restores
# exact hyperparameter gradients. The step is shared; small accessors supply
# the per-type factor, posterior-precision assembly, and rebuild.

_ift_factor(posterior::ChordalGMRF) = posterior.F
_ift_factor(posterior::GMRF) = _mooncake_chordal_factor(posterior.linsolve_cache)
_ift_factor(posterior::WorkspaceGMRF) = _mooncake_workspace_factor(posterior)
_ift_factor(posterior::ConstrainedGMRF) = _mooncake_chordal_factor(posterior.base_gmrf.linsolve_cache)

_raw_mean(d::ChordalGMRF) = d.μ
_raw_mean(d::Union{GMRF, WorkspaceGMRF}) = d.mean
_raw_mean(d::ConstrainedGMRF) = d.base_gmrf.mean

_ift_qpost(prior::ChordalGMRF, H) = hermdiff(precision_matrix(prior), H)
_ift_qpost(prior::Union{GMRF, WorkspaceGMRF, ConstrainedGMRF}, H) = precision_matrix(prior) - H

_ift_rebuild(posterior::ChordalGMRF, x, Q) = ChordalGMRF(x, Q, posterior.F)
_ift_rebuild(posterior::GMRF, x, Q) = _gmrf_with_cache(x, Q, posterior.linsolve_cache)
function _ift_rebuild(posterior::WorkspaceGMRF, x, Q)
    if has_constraints(posterior)
        ci = posterior.constraints
        return WorkspaceGMRF(x, Q, posterior.workspace, ci.matrix, ci.vector)
    end
    return WorkspaceGMRF(x, Q, posterior.workspace)
end
function _ift_rebuild(posterior::ConstrainedGMRF, x, Q)
    base = _gmrf_with_cache(x, Q, posterior.base_gmrf.linsolve_cache)
    return ConstrainedGMRF(base, posterior.constraint_matrix, posterior.constraint_vector)
end

# Project the Newton step onto the constraint tangent space (identity when
# unconstrained). The posterior's precomputed Ã/L_c act as constants —
# justified at convergence exactly like treating the factor as constant. The
# projection also annihilates the KKT-multiplier component of the raw
# gradient, so the corrected iterate stays feasible: A x_corrected = e.
_ift_project(posterior, step) = step
function _ift_project(posterior::WorkspaceGMRF, step)
    has_constraints(posterior) || return step
    ci = posterior.constraints
    return step - _constraint_shift(
        ci.A_tilde_T, ci.L_c, _dense_constraints(ci.matrix) * step
    )
end
function _ift_project(posterior::ConstrainedGMRF, step)
    return step - _constraint_shift(
        posterior.A_tilde_T, posterior.L_c,
        _dense_constraints(posterior.constraint_matrix) * step,
    )
end

function _mooncake_ga_ift(prior, posterior, obslik)
    x_star = mean(posterior)

    # The Newton residual uses the *raw* (unconstrained) prior mean, exactly
    # like the ChainRules GA rules' base_prior: for a constrained prior the
    # difference is a pure KKT-multiplier direction Aᵀu, which the projection
    # annihilates identically — while `constrained_mean` would smuggle in a
    # Q-dependence (∝ the raw-mean constraint residual) that the frozen
    # projection constants cannot account for.
    grad = precision_map(prior) * (x_star .- _raw_mean(prior)) .- loggrad(x_star, obslik)
    step = _ift_project(posterior, _ift_factor(posterior) \ grad)
    x_corrected = x_star - step

    Q_post = _ift_qpost(prior, loghessian(x_corrected, obslik))

    return _ift_rebuild(posterior, x_corrected, Q_post)
end

for P in (:ChordalGMRF, :SparseGMRF, :WorkspaceGMRF, :ConstrainedGMRF)
    @eval @mooncake_overlay function GaussianMarkovRandomFields.gaussian_approximation(
            prior::$P,
            obslik::ObservationLikelihood;
            kwargs...
        )
        # The Gauss–Newton score needs a forward-mode sparse Jacobian that
        # reverse-mode backends cannot differentiate through — same guard as
        # the ChainRules GA rules.
        _has_gauss_newton_jacobian(obslik) && _reverse_mode_gauss_newton_error()
        posterior = gaussian_approximation_notangent(prior, obslik; kwargs...)
        return _mooncake_ga_ift(prior, posterior, obslik)
    end
end

# The conjugate Normal specializations dispatch to their own (more specific)
# methods, so they need their own overlays — which also keeps them unambiguous
# against the generic overlays above. The IFT correction is exact for the
# conjugate case — one Newton step on a quadratic objective — and
# `linear_condition` propagates the prior's algorithm, so the posterior cache
# stays CliqueTrees-backed.
for (P, L) in Iterators.product(
        (:SparseGMRF, :ConstrainedGMRF),
        (:(NormalLikelihood{IdentityLink}), :(LinearlyTransformedLikelihood{<:NormalLikelihood{IdentityLink}})),
    )
    @eval @mooncake_overlay function GaussianMarkovRandomFields.gaussian_approximation(
            prior::$P,
            obslik::$L,
        )
        posterior = gaussian_approximation_notangent(prior, obslik)
        return _mooncake_ga_ift(prior, posterior, obslik)
    end
end
