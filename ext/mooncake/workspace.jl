# `WorkspaceGMRF` (shared-workspace path).
#
# Same architecture again: constructor primitives route tangents to the mean
# and precision snapshot, and the factorization-based operations are overlaid
# with the two-argument Multifrontal forms. Requires the workspace to use the
# `CliqueTreesBackend`. Linear equality constraints are supported: the
# corrections are recomputed differentiably from the shared constraint
# formulas (see constraints.jl), with Q entering through `ldivwith`.

function _require_cliquetrees_backend(ws::GMRFWorkspace)
    ws.backend isa CliqueTreesBackend || throw(
        ArgumentError(
            "Mooncake AD through a WorkspaceGMRF requires the CliqueTrees backend. " *
                "Construct the workspace with `GMRFWorkspace(Q, CliqueTreesBackend)`."
        )
    )
    return nothing
end

"""
    _mooncake_workspace_factor(d::WorkspaceGMRF) -> ChordalCholesky

Load `d`'s precision into its workspace, refactorize if needed, and return a
**copy** of the backend's `ChordalCholesky`. A Mooncake primitive with no
tangent, like [`_mooncake_chordal_factor`](@ref).

The copy is essential: the two-arg `logdet`/`selinv`/`ldiv!` rules read the
factor again in their *pullbacks*, but the shared workspace factor may have
been refactorized at a different precision by then (e.g. a prior and its
posterior sharing one workspace within the same objective). A snapshot pins
the factor the reverse pass sees to the one the forward pass used.
"""
function _mooncake_workspace_factor(d::WorkspaceGMRF)
    _require_cliquetrees_backend(d.workspace)
    ensure_loaded!(d)
    ensure_numeric!(d.workspace)
    return copy(d.workspace.backend.factor)
end

@is_primitive MinimalCtx Tuple{typeof(_mooncake_workspace_factor), WorkspaceGMRF}

function Mooncake.rrule!!(
        ::CoDual{typeof(_mooncake_workspace_factor)},
        cdd::CoDual{<:WorkspaceGMRF},
    )
    d = primal(cdd)
    F = _mooncake_workspace_factor(d)
    # zero_rdata, not NoRData: a constrained WorkspaceGMRF has a non-trivial
    # rdata slot (the Float64 log_constraint_correction field).
    _mooncake_workspace_factor_pullback!!(::Any) = (NoRData(), Mooncake.zero_rdata(d))
    return CoDual(F, NoFData()), _mooncake_workspace_factor_pullback!!
end

# --- Constructor primitives ---

@is_primitive MinimalCtx Tuple{Type{WorkspaceGMRF}, AbstractVector, SparseMatrixCSC}
@is_primitive MinimalCtx Tuple{Type{WorkspaceGMRF}, AbstractVector, SparseMatrixCSC, GMRFWorkspace}

function _workspace_gmrf_rrule_impl(cdμ::CoDual, cdQ::CoDual, args...)
    μ, Σμ = MooncakeSparse.primaltangent(cdμ)
    Q, ΣQ = MooncakeSparse.primaltangent(cdQ)

    gmrf = WorkspaceGMRF(μ, Q, args...)
    # Gate before zero_tangent so a CHOLMOD-backed workspace fails with an
    # actionable message instead of deep inside tangent generation.
    _require_cliquetrees_backend(gmrf.workspace)
    dy = fdata(zero_tangent(gmrf))

    function WorkspaceGMRF_pullback!!(::Any)
        dμ = MooncakeSparse.toarray(gmrf.mean, dy.data.mean)
        dQ = MooncakeSparse.toarray(gmrf.precision, dy.data.precision)

        Σμ .+= dμ
        nonzeros(ΣQ) .+= nonzeros(dQ)

        return ntuple(_ -> NoRData(), 3 + length(args))
    end

    return CoDual(gmrf, dy), WorkspaceGMRF_pullback!!
end

function Mooncake.rrule!!(
        ::CoDual{Type{WorkspaceGMRF}},
        cdμ::CoDual{<:AbstractVector},
        cdQ::CoDual{<:SparseMatrixCSC},
    )
    return _workspace_gmrf_rrule_impl(cdμ, cdQ)
end

function Mooncake.rrule!!(
        ::CoDual{Type{WorkspaceGMRF}},
        cdμ::CoDual{<:AbstractVector},
        cdQ::CoDual{<:SparseMatrixCSC},
        cdws::CoDual{<:GMRFWorkspace},
    )
    return _workspace_gmrf_rrule_impl(cdμ, cdQ, primal(cdws))
end

# Constrained construction. Tangents flow to the mean and precision; the
# ConstraintInfo is treated as derived data, except for `constrained_mean`,
# whose tangent is routed back to μ through the constraint projection
# (μ_c = μ − Ã L_c⁻¹(Aμ − e), so μ̄ += Pᵀ c̄ with Pᵀ = I − Aᵀ L_c⁻¹ Ãᵀ).
# The Q-dependence of μ_c is proportional to the constraint residual and is
# dropped, matching the ConstrainedGMRF ChainRules rule.
@is_primitive MinimalCtx Tuple{
    Type{WorkspaceGMRF}, AbstractVector, SparseMatrixCSC, GMRFWorkspace,
    AbstractMatrix, AbstractVector,
}

function Mooncake.rrule!!(
        ::CoDual{Type{WorkspaceGMRF}},
        cdμ::CoDual{<:AbstractVector},
        cdQ::CoDual{<:SparseMatrixCSC},
        cdws::CoDual{<:GMRFWorkspace},
        cdA::CoDual{<:AbstractMatrix},
        cde::CoDual{<:AbstractVector},
    )
    μ, Σμ = MooncakeSparse.primaltangent(cdμ)
    Q, ΣQ = MooncakeSparse.primaltangent(cdQ)

    gmrf = WorkspaceGMRF(μ, Q, primal(cdws), primal(cdA), primal(cde))
    _require_cliquetrees_backend(gmrf.workspace)
    dy = fdata(zero_tangent(gmrf))

    function WorkspaceGMRF_constrained_pullback!!(::Any)
        dμ = MooncakeSparse.toarray(gmrf.mean, dy.data.mean)
        dQ = MooncakeSparse.toarray(gmrf.precision, dy.data.precision)

        ci = gmrf.constraints
        c̄ = MooncakeSparse.tangentdata(dy.data.constraints).constrained_mean
        v = ci.L_c \ (ci.A_tilde_T' * c̄)

        Σμ .+= dμ .+ c̄ .- ci.matrix' * v
        nonzeros(ΣQ) .+= nonzeros(dQ)

        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(gmrf, dy), WorkspaceGMRF_constrained_pullback!!
end

# --- Factorization-based operations ---

# --- Factorization-based operations ---

@mooncake_overlay function logpdf(d::WorkspaceGMRF, z::AbstractVector)
    F = _mooncake_workspace_factor(d)
    r = z - d.mean
    n = length(r)
    val = (logdet(Symmetric(d.precision), F) - n * log(2π) - dot(r, d.precision, r)) / 2
    if has_constraints(d)
        val += _ws_constraint_correction(d, F)
    end
    return val
end

@mooncake_overlay function logdetcov(d::WorkspaceGMRF)
    F = _mooncake_workspace_factor(d)
    return -logdet(Symmetric(d.precision), F)
end

@mooncake_overlay function var(d::WorkspaceGMRF)
    F = _mooncake_workspace_factor(d)
    σ = diag(Multifrontal.selinv(Symmetric(d.precision), F))
    if has_constraints(d)
        _, A_tilde_T, S_c = _ws_constraint_schur(d, F)
        σ = max.(σ .- _constraint_var_correction(A_tilde_T, S_c), 0.0)
    end
    return σ
end
