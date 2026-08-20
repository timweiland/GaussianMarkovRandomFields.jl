# Reverse-mode `logpdf` for a GMRF.
#
#     ∂logpdf/∂μ = Q(z - μ)
#     ∂logpdf/∂Q = ½(Q⁻¹ - (z-μ)(z-μ)ᵀ)
#     ∂logpdf/∂z = -Q(z - μ)
#
# The `Q⁻¹` term comes from a selected inversion, so it is only ever read on
# `Q`'s stored pattern — see `accumulate_on_pattern!` for why that matters and
# why the previous `x.dval.precision .+= Q̄` was unsound.
#
# A `ConstrainedGMRF` adds a `log_constraint_correction` term on top of its base
# GMRF's `logpdf`. That correction depends on the base `μ` and `Q`, so dropping
# it would leave a plausible-looking but incomplete gradient.

# `logpdf` is evaluated against the *base* GMRF's mean even for a constrained
# GMRF, whose `mean` reports the constrained one.
logpdf_mean(d::AbstractGMRF) = mean(d)
logpdf_mean(d::ConstrainedGMRF) = mean(d.base_gmrf)

"""
    accumulate_constraint_correction!(shadow, d, ȳ)

Add the `log_constraint_correction` term's own `μ` and `Q` cotangents.

They are folded straight into the base GMRF's shadow slots rather than stored as
a cotangent on the scalar field itself, which is not an option: `ConstrainedGMRF`
is an immutable struct, so its shadow's `log_constraint_correction::T` cannot be
accumulated into. Folding is equivalent — the correction is a deterministic
function of the base `μ` and `Q` — and it keeps the whole term on the books.

Formulas match the `ConstrainedGMRF` branch of `_extract_posterior_tangents` in
`src/autodiff/gaussian_approximation.jl`.
"""
accumulate_constraint_correction!(shadow, d::AbstractGMRF, ȳ) = shadow

function accumulate_constraint_correction!(shadow, d::ConstrainedGMRF, ȳ)
    A = d.constraint_matrix
    resid = d.constraint_vector - A * mean(d.base_gmrf)

    # ∂c/∂μ
    shadow_mean(shadow) .+= ȳ .* (A' * (d.L_c \ (-resid)))

    # ∂c/∂Q = -½ Ã (S⁻¹ - w wᵀ) Ãᵀ, dense but read only on Q's stored pattern.
    S_inv = inv(d.L_c)
    w = S_inv * resid
    Q̄_corr = (ȳ * -0.5) .* (d.A_tilde_T * (S_inv - w * w') * d.A_tilde_T')
    accumulate_on_pattern!(
        (i, j) -> Q̄_corr[i, j], shadow_precision(shadow), precision_storage(d)
    )
    return shadow
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(logpdf)},
        ::Type{RT},
        x::Annotation{<:AbstractGMRF},
        z::Annotation{<:AbstractVector}
    ) where {RT}
    primal = logpdf(x.val, z.val)

    if !is_active(x) && !is_active(z)
        tape = nothing
    else
        enzyme_check_supported("logpdf", x.val)
        Q = precision_matrix(x.val)
        r = z.val - logpdf_mean(x.val)
        # Only the precision cotangent needs the selected inverse; a `Const`
        # GMRF differentiated w.r.t. `z` alone gets away without it.
        Σ = is_active(x) ? enzyme_selinv(x.val) : nothing
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

    if is_active(x)
        xs = shadow_of(x)
        shadow_mean(xs) .+= ȳ .* Qr

        half = 0.5 * ȳ
        accumulate_on_pattern!(
            (i, j) -> half * (Σ[i, j] - r[i] * r[j]),
            shadow_precision(xs),
            precision_storage(x.val),
        )

        accumulate_constraint_correction!(xs, x.val, ȳ)
    end

    if is_active(z)
        shadow_of(z) .-= ȳ .* Qr
    end

    return (nothing, nothing)
end
