# Rules whose only job is to stop Mooncake before it reaches CHOLMOD.
#
# Mooncake support in this package is built for `ChordalGMRF`: its factorization
# is pure Julia, so Mooncake can traverse it, and the chordal rules in this
# extension intercept the few places where traversing it is not what we want. A
# plain `GMRF` factorizes through SuiteSparse's CHOLMOD instead, which Mooncake
# only ever sees as `Ptr`s into memory owned by a C library. With no primitive
# covering it, Mooncake descends into `logdet(::CHOLMOD.Factor)` and hands its
# `pointerref` rule one of those pointers — and the process dies with signal 11
# before anything can be raised.
#
# A segfault is worse than a wrong gradient: it cannot be caught, it names no
# line of user code, and under a test runner it takes every other test down with
# it. So each operation that actually consumes a CHOLMOD factorization is
# declared primitive here with a rule that refuses on the spot.
#
# These are guards, not support. Adding genuine `GMRF` support means a
# hand-written rule per solver primitive; the CliqueTrees-backed factorization
# that `ChordalGMRF` already uses is the better route, since it is pure Julia and
# therefore traversable.
#
# Each guard dispatches on `CHOLMODFactorization` specifically, so it is exactly
# the algorithm that crashes that gets refused. The other algorithms this package
# resolves to — `LDLtFactorization`, `DiagonalFactorization` — are pure Julia and
# are left alone. Guarding the `*_impl` methods rather than `logdetcov`/`logpdf`
# also means the `DefaultLinearSolver` indirection is covered for free: it
# resolves to the concrete algorithm and then calls straight back into these.

"""
    mooncake_supported_routes() -> String

The way out, shared by every refusal in this extension so that a user meets the
same three options wherever they hit one.
"""
function mooncake_supported_routes()
    return """
    Mooncake support rests on CliqueTrees' pure-Julia factorization. Any of:

        GMRF(μ, Q, LinearSolve.CliqueTreesFactorization())
        GMRFWorkspace(Q, CliqueTreesBackend)     # for a WorkspaceGMRF
        ChordalGMRF(μ, Q)

    ForwardDiff handles every GMRF type. The per-backend support matrix is in \
    the "Automatic Differentiation" reference page of the documentation.
    """
end

"""
    mooncake_unsupported(op)

Refuse to differentiate `op` for a CHOLMOD-backed GMRF rather than letting
Mooncake walk into CHOLMOD, where it segfaults the process. `op` arrives
pre-formatted, so that a guard covering two operations can name both.
"""
@noinline function mooncake_unsupported(op)
    return throw(
        ArgumentError(
            """
            Mooncake cannot differentiate $op for a CHOLMOD-backed GMRF.

            Reaching it means differentiating through SuiteSparse's CHOLMOD, \
            which Mooncake sees only as pointers into memory owned by a C \
            library. Left to itself it dereferences one and takes the whole \
            process down with a segmentation fault, so this package refuses \
            first.

            $(mooncake_supported_routes())
            """
        )
    )
end

"""
    mooncake_wrong_factorization(what)

Refuse an operation whose GMRF reached it carrying a factorization Mooncake has
no rules for. Distinct from [`mooncake_unsupported`](@ref): that one guards the
solver entry points a CHOLMOD-backed GMRF reaches, this one fires earlier, in
the overlays, where the factorization is inspected directly.
"""
@noinline function mooncake_wrong_factorization(what)
    return throw(
        ArgumentError(
            """
            Mooncake AD through $what requires the CliqueTrees factorization.

            $(mooncake_supported_routes())
            """
        )
    )
end

# `logdetcov`, and with it `logpdf`, which decomposes into `logdetcov + sqmahal`.
@is_primitive MinimalCtx Tuple{typeof(_logdet_cov_impl), Any, CHOLMODFactorization}

function Mooncake.rrule!!(
        ::CoDual{typeof(_logdet_cov_impl)},
        ::CoDual,
        ::CoDual{<:CHOLMODFactorization},
    )
    return mooncake_unsupported("`logdetcov`")
end

# `var` and `std`, which read marginal variances off a selected inversion.
@is_primitive MinimalCtx Tuple{typeof(_selinv_diag_impl), Any, CHOLMODFactorization}

function Mooncake.rrule!!(
        ::CoDual{typeof(_selinv_diag_impl)},
        ::CoDual,
        ::CoDual{<:CHOLMODFactorization},
    )
    return mooncake_unsupported("`var`/`std`")
end

# The full selected inverse, reached through the exported `selinv`.
@is_primitive MinimalCtx Tuple{typeof(_selinv_impl), Any, CHOLMODFactorization}

function Mooncake.rrule!!(
        ::CoDual{typeof(_selinv_impl)},
        ::CoDual,
        ::CoDual{<:CHOLMODFactorization},
    )
    return mooncake_unsupported("`selinv`")
end

# The triangular solve `L' \ x`, reached through the exported `backward_solve`.
# `rand` is built on it too, but Mooncake never gets that far: it gives up on
# `_rand_impl!` first, with a `BoundsError` of its own from inside the rule
# compiler. That is already a loud failure, so it is left alone.
@is_primitive MinimalCtx Tuple{typeof(_backward_solve_impl), Any, Any, CHOLMODFactorization}

function Mooncake.rrule!!(
        ::CoDual{typeof(_backward_solve_impl)},
        ::CoDual,
        ::CoDual,
        ::CoDual{<:CHOLMODFactorization},
    )
    return mooncake_unsupported("`backward_solve`")
end
