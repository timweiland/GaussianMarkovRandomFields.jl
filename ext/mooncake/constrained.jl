# `ConstrainedGMRF`: hard linear equality constraints over a CliqueTrees GMRF.
#
# Same treatment as the constrained `WorkspaceGMRF`: the constructor primitive
# routes mean/precision tangents into the base GMRF (plus the constrained-mean
# projection), and `logpdf`/`var` recompute the corrections differentiably from
# the shared constraint formulas (see constraints.jl).

@is_primitive MinimalCtx Tuple{Type{ConstrainedGMRF}, GMRF, AbstractMatrix, AbstractVector}

function Mooncake.rrule!!(
        ::CoDual{Type{ConstrainedGMRF}},
        cdbase::CoDual{<:GMRF},
        cdA::CoDual{<:AbstractMatrix},
        cde::CoDual{<:AbstractVector},
    )
    base = primal(cdbase)
    dbase_in = MooncakeSparse.tangentdata(tangent(cdbase))

    cgmrf = ConstrainedGMRF(base, primal(cdA), primal(cde))
    # Gate before zero_tangent so a non-CliqueTrees base fails with an
    # actionable message instead of deep inside tangent generation.
    _mooncake_chordal_factor(base.linsolve_cache)
    dy = fdata(zero_tangent(cgmrf))

    function ConstrainedGMRF_pullback!!(::Any)
        dyb = MooncakeSparse.tangentdata(dy.data.base_gmrf)
        dμ = MooncakeSparse.toarray(base.mean, dyb.mean)
        dQ = MooncakeSparse.toarray(base.precision, dyb.precision)

        # constrained_mean tangent: routed back to the base mean through the
        # projection Pᵀ = I − Aᵀ L_c⁻¹ Ãᵀ (Q-dependence ∝ constraint residual
        # dropped, matching the ChainRules ConstrainedGMRF rule).
        c̄ = MooncakeSparse.toarray(cgmrf.constrained_mean, dy.data.constrained_mean)
        v = cgmrf.L_c \ (cgmrf.A_tilde_T' * c̄)

        MooncakeSparse.toarray(base.mean, dbase_in.mean) .+= dμ .+ c̄ .- cgmrf.constraint_matrix' * v
        nonzeros(MooncakeSparse.toarray(base.precision, dbase_in.precision)) .+= nonzeros(dQ)

        return NoRData(), Mooncake.zero_rdata(base), NoRData(), NoRData()
    end

    return CoDual(cgmrf, dy), ConstrainedGMRF_pullback!!
end

function _constrained_gmrf_schur(d::ConstrainedGMRF)
    base = d.base_gmrf
    F = _mooncake_chordal_factor(base.linsolve_cache)
    return _mooncake_constraint_schur(precision_map(base), F, d.constraint_matrix)
end

@mooncake_overlay function logpdf(d::ConstrainedGMRF, z::AbstractVector)
    A_dense, _, S_c = _constrained_gmrf_schur(d)
    correction = _constraint_log_correction(
        A_dense, d.constraint_vector, d.base_gmrf.mean, S_c
    )
    return logpdf(d.base_gmrf, z) + correction
end

@mooncake_overlay function var(d::ConstrainedGMRF)
    σ = var(d.base_gmrf)
    _, A_tilde_T, S_c = _constrained_gmrf_schur(d)
    return max.(σ .- _constraint_var_correction(A_tilde_T, S_c), 0.0)
end
