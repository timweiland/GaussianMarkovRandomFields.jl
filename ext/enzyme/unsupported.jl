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
    d isa Const || enzyme_unsupported("var", d.val)

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

# --- gaussian_approximation -------------------------------------------------
#
# This one *did* have an Implicit-Function-Theorem rule, and it was silently
# wrong: on Julia 1.10 the whole pipeline returned [0.84, -120.55] where the
# true gradient is [-214.07, -178.58], and on 1.12 it corrupted a sparse shadow
# badly enough to raise from inside `SparseMatrixCSC`'s constructor.
#
# The reason it is refused rather than repaired is that the IFT rule has to move
# precision cotangents between the prior and the approximation posterior, and
# those two do not agree on how a precision is stored — a prior built from a
# user's `Q` is a bare `SparseMatrixCSC` where every entry is its own variable,
# while the posterior comes back as `Symmetric(Q, :U)` where only one triangle is
# read and its off-diagonals count double. Reconciling that correctly across all
# three GMRF types is a larger piece of work than this fix, and a rule that gets
# it half right is exactly the failure mode being removed here.
#
# Zygote, Mooncake and ForwardDiff all have working `gaussian_approximation`
# rules; the error message points at them.

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(gaussian_approximation)},
        ::Type{RT},
        prior_gmrf::Annotation,
        obs_lik::Annotation
    ) where {RT}
    (prior_gmrf isa Const && obs_lik isa Const) ||
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
