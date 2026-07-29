# Reverse-mode `logpdf` for a GMRF.
#
#     ∂logpdf/∂μ = Q(z - μ)
#     ∂logpdf/∂Q = ½(Q⁻¹ - (z-μ)(z-μ)ᵀ)
#     ∂logpdf/∂z = -Q(z - μ)
#
# The `Q⁻¹` term comes from a selected inversion, so it is only ever read on
# `Q`'s stored pattern — see `accumulate_on_pattern!` for why that matters and
# why the previous `x.dval.precision .+= Q̄` was unsound.

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(logpdf)},
        ::Type{RT},
        x::Annotation{<:AbstractGMRF},
        z::Annotation{<:AbstractVector}
    ) where {RT}
    primal = logpdf(x.val, z.val)

    if x isa Const && z isa Const
        tape = nothing
    else
        enzyme_check_supported("logpdf", x.val)
        Q = precision_matrix(x.val)
        r = z.val - mean(x.val)
        # Only the precision cotangent needs the selected inverse; a `Const`
        # GMRF differentiated w.r.t. `z` alone gets away without it.
        Σ = x isa Const ? nothing : enzyme_selinv(x.val)
        tape = (Q, r, Σ)
    end

    primal_out = EnzymeRules.needs_primal(config) ? primal : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(typeof(primal)) : nothing

    return EnzymeRules.AugmentedReturn(primal_out, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(logpdf)},
        dret::Active,
        tape,
        x::Annotation{<:AbstractGMRF},
        z::Annotation{<:AbstractVector}
    )
    tape === nothing && return (nothing, nothing)

    Q, r, Σ = tape
    ȳ = dret.val
    Qr = Q * r                                  # shared by the μ and z gradients

    if x isa Duplicated
        shadow_mean(x.dval) .+= ȳ .* Qr

        half = 0.5 * ȳ
        accumulate_on_pattern!(
            (i, j) -> half * (Σ[i, j] - r[i] * r[j]),
            shadow_precision(x.dval),
            precision_storage(x.val),
        )
    end

    if z isa Duplicated
        z.dval .-= ȳ .* Qr
    end

    return (nothing, nothing)
end
