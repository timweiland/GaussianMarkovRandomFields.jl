# COV_EXCL_START
# `logdetcov` had no Enzyme rule at all, so Enzyme differentiated the underlying
# factorization directly. For a `GMRF` that means CHOLMOD's `ccall`s, which carry
# no derivative information — Enzyme walked them and returned `[0.0, 0.0]`, a
# silent zero on every Julia version tested. For a `ChordalGMRF` it means the
# pure-Julia multifrontal factorization, which Enzyme *can* walk and got wrong by
# two orders of magnitude. Both are the dangerous kind of failure: a plausible
# number that never raises.
#
# The rule below sidesteps the factorization entirely.
#
#     logdetcov(d) = -logdet(Q)      =>      ∂/∂Q = -Q⁻¹
#
# and only the entries of `Q⁻¹` sitting on `Q`'s stored pattern are needed, which
# is exactly what a selected inversion computes.

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(logdetcov)},
        ::Type{RT},
        x::Annotation{<:AbstractGMRF}
    ) where {RT}
    primal = logdetcov(x.val)

    # The selected inverse is the whole reverse pass; skip it when the GMRF is
    # `Const`, since then there is nowhere to accumulate into.
    if is_active(x)
        enzyme_check_supported("logdetcov", x.val)
        tape = enzyme_selinv(x.val)
    else
        tape = nothing
    end

    primal_out = EnzymeRules.needs_primal(config) ? primal : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(typeof(primal)) : nothing

    return EnzymeRules.AugmentedReturn(primal_out, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(logdetcov)},
        dret::Active,
        tape,
        x::Annotation{<:AbstractGMRF}
    )
    if is_active(x)
        Σ = tape
        ȳ = dret.val
        accumulate_on_pattern!(
            (i, j) -> -ȳ * Σ[i, j],
            shadow_precision(shadow_of(x)),
            precision_storage(x.val),
        )
    end
    return (nothing,)
end
# COV_EXCL_STOP
