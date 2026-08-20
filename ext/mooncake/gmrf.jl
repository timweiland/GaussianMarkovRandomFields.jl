# CliqueTrees-backed `GMRF` (`LinearSolve.CliqueTreesFactorization`).
#
# Same architecture as `ChordalGMRF`: constructor primitives stop Mooncake at
# the LinearSolve-cache boundary, and the cache-based operations are overlaid
# with the two-argument `logdet(A, F)` / `selinv(A, F)` forms from
# CliqueTrees.Multifrontal, whose Mooncake rules propagate gradients to `A`
# while treating the factorization as constant.

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
    # A CHOLMOD-backed GMRF reaches this before it reaches the solver-level
    # guards in unsupported.jl, since `logdetcov`/`var` are overlaid here.
    F isa ChordalCholesky || mooncake_wrong_factorization("a GMRF")
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
