module GaussianMarkovRandomFieldsMooncake

using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: hermdiff, ensure_factorization!,
    ensure_loaded!, ensure_numeric!, has_constraints,
    _has_gauss_newton_jacobian, _reverse_mode_gauss_newton_error,
    _constraint_shift, _constraint_log_correction, _constraint_var_correction
using Statistics: mean
using Distributions: logpdf, logdetcov, var
using Mooncake
using Mooncake: @is_primitive, @mooncake_overlay, MinimalCtx, CoDual, NoRData, NoFData, primal, tangent, fdata, zero_tangent
using MooncakeSparse
using SparseArrays: nonzeros, SparseMatrixCSC
using LinearAlgebra: Hermitian, Symmetric, cholesky, diag, dot, logdet, I
using LinearSolve
using CliqueTrees.Multifrontal: ChordalCholesky
import CliqueTrees.Multifrontal as Multifrontal

# A GMRF whose precision is a plain sparse matrix — the shape produced by the
# CliqueTrees LinearSolve backend. The overlays below additionally require the
# linsolve cache to hold a `ChordalCholesky` (enforced at runtime with an
# actionable error), so other backends fail loudly instead of deep in AD.
const SparseGMRF = GMRF{<:Real, <:AbstractVector, <:Any, <:SparseMatrixCSC}

@is_primitive MinimalCtx Tuple{Type{ChordalGMRF}, AbstractVector, SparseMatrixCSC}

function Mooncake.rrule!!(
        ::CoDual{Type{ChordalGMRF}},
        cdμ::CoDual{<:AbstractVector},
        cdQ::CoDual{<:SparseMatrixCSC},
    )
    μ, Σμ = MooncakeSparse.primaltangent(cdμ)
    Q, ΣQ = MooncakeSparse.primaltangent(cdQ)

    gmrf = ChordalGMRF(μ, Q)
    dy = fdata(zero_tangent(gmrf))

    function pullback!!(::NoRData)
        dμ = MooncakeSparse.toarray(gmrf.μ, dy.data.μ)
        dQ = MooncakeSparse.toarray(gmrf.Q, dy.data.Q)

        Σμ .+= dμ
        nonzeros(ΣQ) .+= nonzeros(parent(dQ))

        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(gmrf, dy), pullback!!
end

@is_primitive MinimalCtx Tuple{Type{ChordalGMRF}, AbstractVector, Hermitian, ChordalCholesky}

function Mooncake.rrule!!(
        ::CoDual{Type{ChordalGMRF}},
        cdμ::CoDual{<:AbstractVector},
        cdQ::CoDual{<:Hermitian},
        cdF::CoDual{<:ChordalCholesky},
    )
    μ, Σμ = MooncakeSparse.primaltangent(cdμ)
    Q, ΣQ = MooncakeSparse.primaltangent(cdQ)
    F = primal(cdF)

    gmrf = ChordalGMRF(μ, Q, F)
    dy = fdata(zero_tangent(gmrf))

    function pullback!!(::NoRData)
        dμ = MooncakeSparse.toarray(gmrf.μ, dy.data.μ)
        dQ = MooncakeSparse.toarray(gmrf.Q, dy.data.Q)

        Σμ .+= dμ
        nonzeros(parent(ΣQ)) .+= nonzeros(parent(dQ))

        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(gmrf, dy), pullback!!
end

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

# (The gaussian_approximation overlays for all prior types live in the shared
# IFT section at the end of this file.)

# =============================================================================
# CliqueTrees-backed GMRF (LinearSolve.CliqueTreesFactorization)
#
# Same architecture as ChordalGMRF above: constructor primitives stop Mooncake
# at the LinearSolve-cache boundary, and the cache-based operations are
# overlaid with the two-arg `logdet(A, F)` / `selinv(A, F)` forms from
# CliqueTrees.Multifrontal, whose Mooncake rules propagate gradients to `A`
# while treating the factorization as constant.
# =============================================================================

"""
    _mooncake_chordal_factor(cache::LinearSolve.LinearCache) -> ChordalCholesky

Return the (numerically up-to-date) `ChordalCholesky` held by a
CliqueTrees-backed LinearSolve cache. A Mooncake primitive with no tangent:
the factor only acts as a solver, gradients flow through the two-arg
`logdet`/`selinv`/`ldiv!` rules that reference the precision directly.
"""
function _mooncake_chordal_factor(cache::LinearSolve.LinearCache)
    ensure_factorization!(cache)
    F = cache.cacheval
    F isa ChordalCholesky || throw(
        ArgumentError(
            "Mooncake AD through a GMRF requires the CliqueTrees backend. " *
                "Construct the GMRF with `GMRF(μ, Q, LinearSolve.CliqueTreesFactorization())`."
        )
    )
    return F
end

@is_primitive MinimalCtx Tuple{typeof(_mooncake_chordal_factor), LinearSolve.LinearCache}

function Mooncake.rrule!!(
        ::CoDual{typeof(_mooncake_chordal_factor)},
        cdcache::CoDual{<:LinearSolve.LinearCache},
    )
    F = _mooncake_chordal_factor(primal(cdcache))
    _mooncake_chordal_factor_pullback!!(::NoRData) = (NoRData(), NoRData())
    return CoDual(F, NoFData()), _mooncake_chordal_factor_pullback!!
end

# --- Constructor primitives ---
# Tangents flow to the mean and precision; the LinearSolve cache (and its
# ChordalCholesky cacheval, tangent-free by design) is non-differentiable.

@is_primitive MinimalCtx Tuple{Type{GMRF}, AbstractVector, SparseMatrixCSC, LinearSolve.CliqueTreesFactorization}

function Mooncake.rrule!!(
        ::CoDual{Type{GMRF}},
        cdμ::CoDual{<:AbstractVector},
        cdQ::CoDual{<:SparseMatrixCSC},
        cdalg::CoDual{<:LinearSolve.CliqueTreesFactorization},
    )
    μ, Σμ = MooncakeSparse.primaltangent(cdμ)
    Q, ΣQ = MooncakeSparse.primaltangent(cdQ)

    gmrf = GMRF(μ, Q, primal(cdalg))
    dy = fdata(zero_tangent(gmrf))

    # The incoming rdata is ignored: mean and precision tangents live in the
    # (shared) fdata, and the only rdata component is the non-differentiable
    # RNG state inside the default RBMC strategy.
    function GMRF_pullback!!(::Any)
        dμ = MooncakeSparse.toarray(gmrf.mean, dy.data.mean)
        dQ = MooncakeSparse.toarray(gmrf.precision, dy.data.precision)

        Σμ .+= dμ
        nonzeros(ΣQ) .+= nonzeros(dQ)

        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(gmrf, dy), GMRF_pullback!!
end

# Cache-reusing constructor for the gaussian_approximation overlay below:
# rebuilds a GMRF around an existing (already factorized) cache without
# triggering a new symbolic factorization.
function _gmrf_with_cache(μ::AbstractVector, Q::SparseMatrixCSC, cache::LinearSolve.LinearCache)
    return GMRF(μ, Q; linsolve_cache = cache)
end

@is_primitive MinimalCtx Tuple{typeof(_gmrf_with_cache), AbstractVector, SparseMatrixCSC, LinearSolve.LinearCache}

function Mooncake.rrule!!(
        ::CoDual{typeof(_gmrf_with_cache)},
        cdμ::CoDual{<:AbstractVector},
        cdQ::CoDual{<:SparseMatrixCSC},
        cdcache::CoDual{<:LinearSolve.LinearCache},
    )
    μ, Σμ = MooncakeSparse.primaltangent(cdμ)
    Q, ΣQ = MooncakeSparse.primaltangent(cdQ)

    gmrf = _gmrf_with_cache(μ, Q, primal(cdcache))
    dy = fdata(zero_tangent(gmrf))

    function _gmrf_with_cache_pullback!!(::Any)
        dμ = MooncakeSparse.toarray(gmrf.mean, dy.data.mean)
        dQ = MooncakeSparse.toarray(gmrf.precision, dy.data.precision)

        Σμ .+= dμ
        nonzeros(ΣQ) .+= nonzeros(dQ)

        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(gmrf, dy), _gmrf_with_cache_pullback!!
end

# --- Cache-based operations, rerouted to differentiable two-arg forms ---
# `logpdf` needs no rule of its own: the Distributions fallback decomposes
# into `logdetcov` (overlaid here) and `sqmahal`/`gradlogpdf`, which Mooncake
# traces natively via the MooncakeSparse mul/dot rules.

@mooncake_overlay function logdetcov(d::SparseGMRF)
    F = _mooncake_chordal_factor(d.linsolve_cache)
    return -logdet(Symmetric(precision_map(d)), F)
end

@mooncake_overlay function var(d::SparseGMRF)
    F = _mooncake_chordal_factor(d.linsolve_cache)
    Σ = Multifrontal.selinv(Symmetric(precision_map(d)), F)
    return diag(Σ)
end

# =============================================================================
# WorkspaceGMRF (shared-workspace path)
#
# Same architecture again: constructor primitives route tangents to the mean
# and precision snapshot, and the factorization-based operations are overlaid
# with the two-arg Multifrontal forms. Requires the workspace to use the
# `CliqueTreesBackend`. Linear equality constraints are supported: the
# corrections are recomputed differentiably from the shared constraint
# formulas, with Q entering through `ldivwith`.
# =============================================================================

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

# Differentiable recomputation of the constraint Schur quantities that
# `ConstraintInfo`/`ConstrainedGMRF` cache: the precision enters through
# Ã = Q⁻¹Aᵀ, computed with `ldivwith` (the factor snapshot is only the
# solver, gradients flow to Q), and the small m×m Schur complement uses
# Mooncake's dense Cholesky rules. The corrections themselves come from the
# shared `_constraint_*` formulas.
function _mooncake_constraint_schur(Q::SparseMatrixCSC, F::ChordalCholesky, A::AbstractMatrix)
    A_tilde_T = MooncakeSparse.ldivwith(Symmetric(Q), F, Matrix(A'))
    S_c = cholesky(Symmetric(A * A_tilde_T))
    return A_tilde_T, S_c
end

function _ws_constraint_schur(d::WorkspaceGMRF, F::ChordalCholesky)
    return _mooncake_constraint_schur(d.precision, F, d.constraints.matrix)
end

function _ws_constraint_correction(d::WorkspaceGMRF, F::ChordalCholesky)
    ci = d.constraints
    _, S_c = _ws_constraint_schur(d, F)
    return _constraint_log_correction(ci.matrix, ci.vector, d.mean, S_c)
end

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
        A_tilde_T, S_c = _ws_constraint_schur(d, F)
        σ = max.(σ .- _constraint_var_correction(A_tilde_T, S_c), 0.0)
    end
    return σ
end

# =============================================================================
# ConstrainedGMRF (hard linear equality constraints over a CliqueTrees GMRF)
#
# Same treatment as the constrained WorkspaceGMRF: the constructor primitive
# routes mean/precision tangents into the base GMRF (plus the constrained-mean
# projection), and logpdf/var recompute the corrections differentiably from
# the shared constraint formulas.
# =============================================================================

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
    _, S_c = _constrained_gmrf_schur(d)
    correction = _constraint_log_correction(
        d.constraint_matrix, d.constraint_vector, d.base_gmrf.mean, S_c
    )
    return logpdf(d.base_gmrf, z) + correction
end

@mooncake_overlay function var(d::ConstrainedGMRF)
    σ = var(d.base_gmrf)
    A_tilde_T, S_c = _constrained_gmrf_schur(d)
    return max.(σ .- _constraint_var_correction(A_tilde_T, S_c), 0.0)
end

# =============================================================================
# IFT-corrected gaussian_approximation (all prior types)
#
# One differentiable Newton step at the converged, tangent-free mode restores
# exact hyperparameter gradients. The step is shared; small accessors supply
# the per-type factor, posterior-precision assembly, and rebuild.
# =============================================================================

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
    return step - _constraint_shift(ci.A_tilde_T, ci.L_c, ci.matrix * step)
end
function _ift_project(posterior::ConstrainedGMRF, step)
    return step - _constraint_shift(
        posterior.A_tilde_T, posterior.L_c, posterior.constraint_matrix * step
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

end
