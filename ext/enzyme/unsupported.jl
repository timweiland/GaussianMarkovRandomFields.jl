# Rules whose only job is to stop Enzyme.
#
# Without them Enzyme walks into the sparse factorization and produces something
# — an `IllegalTypeAnalysisException` if we are lucky, a plausible wrong number
# if we are not. A rule that refuses is strictly better than either.
#
# All of these are no-ops when the GMRF is `Const`: nothing is being
# differentiated with respect to it, so the derivative really is zero and there
# is no reason to complain.

# --- var / std --------------------------------------------------------------
#
# `var(d) = diag(Q⁻¹)` has a perfectly good derivative on paper,
# ∂(Σᵢᵢ)/∂Q = -(Σ ᵉᵢ ᵉᵢᵀ Σ), but evaluating it needs *full rows* of Σ rather than
# the selected inverse, so there is no cheap sparse rule to write. Enzyme's own
# attempt fails with `IllegalTypeAnalysisException` on every Julia version tested.

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(var)},
        ::Type{RT},
        d::Annotation{<:AbstractGMRF}
    ) where {RT}
    is_active(d) && enzyme_unsupported("var", d.val)

    primal = var(d.val)
    primal_out = EnzymeRules.needs_primal(config) ? primal : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(primal) : nothing

    return EnzymeRules.AugmentedReturn(primal_out, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(var)},
        dret,
        tape,
        d::Annotation{<:AbstractGMRF}
    )
    return (nothing,)
end

# --- gaussian_approximation for priors without an IFT rule ------------------
#
# `gaussian_approximation.jl` covers every prior type the *primal*
# `gaussian_approximation` currently accepts, so nothing reaches this fallback
# today — it is excluded from coverage on that basis rather than because it is
# untested. It earns its place by making the next GMRF type someone adds refuse
# loudly instead of silently returning a wrong gradient, which is the failure
# mode this whole extension was rewritten to remove.
# COV_EXCL_START

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(gaussian_approximation)},
        ::Type{RT},
        prior_gmrf::Annotation,
        obs_lik::Annotation
    ) where {RT}
    (is_active(prior_gmrf) || is_active(obs_lik)) &&
        enzyme_unsupported("gaussian_approximation", prior_gmrf.val)

    primal = gaussian_approximation(prior_gmrf.val, obs_lik.val)
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(gaussian_approximation)},
        ::Type{RT},
        tape,
        prior_gmrf::Annotation,
        obs_lik::Annotation
    ) where {RT}
    return (nothing, nothing)
end
# COV_EXCL_STOP
