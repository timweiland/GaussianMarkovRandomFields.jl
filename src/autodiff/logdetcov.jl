using ChainRulesCore
using LinearAlgebra

# Reverse-mode rules for `logdetcov`.
#
# Without these, Zygote differentiates `logdetcov` by tracing into the Cholesky
# factorization stored on the GMRF. That factorization is built outside the
# differentiated code (it is a `NoTangent()` field of the struct, and CHOLMOD's
# symbolic phase is opaque to Zygote), so the log-determinant term contributes a
# *zero* cotangent to `Q` instead of failing. `logpdf` decomposes into
# `logdetcov + sqmahal`, so the result was a silently wrong gradient: correct in
# the `sqmahal` part, missing the log-determinant part entirely.
#
# ∂logdetcov/∂Q = -Q⁻¹. Only the entries of Q⁻¹ on Q's own sparsity pattern are
# needed, because the cotangent is ultimately contracted against a `dQ` that
# lives on that pattern. The selected inverse therefore gives an *exact*
# gradient here, not an approximation — unlike `∂diag(Q⁻¹)/∂Q`, which genuinely
# needs entries of Q⁻¹ outside the pattern and is why `var` has no rule.

"""
    ChainRulesCore.rrule(::typeof(logdetcov), d::ChordalGMRF)

Reverse-mode rule for the log-determinant of a `ChordalGMRF`'s covariance.

The pullback uses the selected inverse from the chordal Cholesky factorization
to form `∂logdetcov/∂Q = -Q⁻¹` restricted to `Q`'s sparsity pattern.
"""
function ChainRulesCore.rrule(::typeof(logdetcov), d::ChordalGMRF)
    val = logdetcov(d)

    function chordal_logdetcov_pullback(ȳ)
        c = -unthunk(ȳ)
        # Skip the selected inversion entirely when the result is unused.
        c isa AbstractZero && return NoTangent(), ZeroTangent()

        # Σ on tril(Q)'s pattern, wrapped as Hermitian so that indexing the
        # upper triangle mirrors the lower one.
        Σ = mselinv(d.Q, d.F)
        Q̄ = Symmetric(c * parent(Σ), Symbol(Σ.uplo))

        x̄ = Tangent{typeof(d)}(
            μ = ZeroTangent(),
            Q = Q̄,
            F = NoTangent(),
        )
        return NoTangent(), x̄
    end

    return val, chordal_logdetcov_pullback
end

"""
    ChainRulesCore.rrule(::typeof(logdetcov), d::GMRF)

Reverse-mode rule for the log-determinant of a `GMRF`'s covariance.

Requires a linear solver that supports selected inversion; the same requirement
as the `logpdf` rule.
"""
function ChainRulesCore.rrule(::typeof(logdetcov), d::GMRF)
    if supports_selinv(d.linsolve_cache.alg) != Val{true}()
        # COV_EXCL_START
        throw(
            ArgumentError(
                "Automatic differentiation through logdetcov requires an algorithm that supports selected inversion " *
                    "(CHOLMODFactorization, CholeskyFactorization). " *
                    "Got algorithm type: $(typeof(d.linsolve_cache.alg)). " *
                    "Consider using LinearSolve.CHOLMODFactorization() or LinearSolve.CholeskyFactorization()."
            )
        )
        # COV_EXCL_STOP
    end

    val = logdetcov(d)

    function gmrf_logdetcov_pullback(ȳ)
        c = -unthunk(ȳ)
        # Skip the selected inversion entirely when the result is unused.
        c isa AbstractZero && return NoTangent(), ZeroTangent()

        Q̄ = c * selinv(d.linsolve_cache)

        x̄ = Tangent{typeof(d)}(
            linsolve_cache = NoTangent(),
            mean = ZeroTangent(),
            precision = Q̄,
            information_vector = NoTangent(),
            Q_sqrt = NoTangent(),
            rbmc_strategy = NoTangent(),
        )
        return NoTangent(), x̄
    end

    return val, gmrf_logdetcov_pullback
end
