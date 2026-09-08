# Mooncake regression coverage for CHOLMOD-backed GMRFs.
#
# Differentiating a plain `GMRF` under Mooncake used to take the entire Julia
# process down with signal 11. Nothing declared a primitive for it, so Mooncake
# descended into `logdet(::CHOLMOD.Factor)` and handed its `pointerref` rule a
# pointer into memory owned by SuiteSparse. Every operation that consumes the
# factorization did it — `logdetcov`, `logpdf`, `gaussian_approximation`, `var` —
# and so did a `GMRF` built *outside* the differentiated function and merely read
# inside it.
#
# The stakes are higher here than for an ordinary regression test: a segfault
# would not fail the suite, it would kill it, and every test after this file
# would silently never run.
#
# ## Why a tridiagonal precision, when the AD reference page asks for fill-in
#
# That advice is aimed at gradient-*correctness* tests, which can silently pass
# on a precision whose factor has no fill-in. Nothing here is such a test: the
# `GMRF` testsets assert that an error is raised, where fill-in cannot matter,
# and the `ChordalGMRF` testset is a contrast check that the guard does not
# over-fire. The fill-in-sensitive chordal gradients live next door, in
# `test_gaussian_approximation_chordal.jl`'s "2D grid precision" testset.
#
# The fixture is tridiagonal for a concrete reason. On Julia 1.10, Mooncake
# cannot build a rule for a `GMRF` over a 2D grid precision at all — it dies in
# its own compiler with `Core.Compiler.KeyError(0)` — so the guard never runs and
# these tests would assert nothing on the LTS. A tridiagonal precision reaches
# the guard on 1.10 and 1.12 alike.

using GaussianMarkovRandomFields
using Distributions: logpdf, logdetcov, Poisson
using SparseArrays
using LinearAlgebra
using LinearSolve
using Random
using Statistics: var
using DifferentiationInterface
using FiniteDiff, Mooncake, MooncakeSparse
using ReTest: @testset, @test

const MOONCAKE = AutoMooncake()
const MOONCAKE_N = 20

"AR(1) precision matrix of size `n` with correlation `ρ`."
function mooncake_ar_precision(ρ, n)
    return spdiagm(-1 => -ρ * ones(n - 1), 0 => ones(n) .+ ρ^2, 1 => -ρ * ones(n - 1))
end

# Top level rather than closures inside the testset: on Julia 1.10 a local
# function captured into a `@testset for` loop header reaches Mooncake as an
# undefined binding, and the test then fails for a reason unrelated to what it
# is testing.
mooncake_Q(θ) = exp(θ[1]) * mooncake_ar_precision(0.5, MOONCAKE_N)
mooncake_μ(θ) = θ[2] * ones(MOONCAKE_N)

# Naming the algorithm keeps the guard's dispatch honest: this is the CHOLMOD
# path, not whatever `DefaultLinearSolver` happens to resolve to.
mooncake_cholmod_gmrf(θ) =
    GMRF(mooncake_μ(θ), mooncake_Q(θ), LinearSolve.CHOLMODFactorization())
# The same precision reached through `DefaultLinearSolver`, which is what the
# two-argument constructor gives you and what the original crash report used.
# The guards sit on the `*_impl` methods, below that indirection, so this has to
# be refused as well.
mooncake_default_gmrf(θ) = GMRF(mooncake_μ(θ), mooncake_Q(θ))
mooncake_chordal_gmrf(θ) = ChordalGMRF(mooncake_μ(θ), mooncake_Q(θ))

"""
Is `e` this package's refusal, as opposed to an error Mooncake raised itself?

The exception *type* cannot answer that. Mooncake throws a bare `ArgumentError`
of its own when it meets a `CHOLMOD.Factor` tangent it cannot convert
(`to_cr_tangent`), so a test that only checks `isa ArgumentError` would read that
as the guard firing. The message is what distinguishes them: ours always names
the GMRF type that works.
"""
mooncake_refusal(e) = e isa ArgumentError && occursin("ChordalGMRF", sprint(showerror, e))

@testset "Mooncake AD support" begin
    Random.seed!(42)

    n = MOONCAKE_N
    θ = [log(2.0), 0.3]
    z = randn(n)

    @testset "GMRF raises rather than segfaulting" begin
        obs_lik = ExponentialFamily(Poisson)(PoissonObservations(rand(1:4, n)))

        # One entry per operation a user would actually write that must not be
        # allowed to reach CHOLMOD. Which guard catches it varies: `logdetcov`,
        # `logpdf` and `var` are overlaid for the CliqueTrees backend and refuse
        # there on inspecting the factorization, while `selinv` and
        # `backward_solve` have no overlay and reach the solver-level primitives
        # in `ext/mooncake/unsupported.jl`. Both refusals carry the same message
        # body, which is what the assertions below pin. `rand` is absent on purpose:
        # it is built on `backward_solve`, but Mooncake gives up on `_rand_impl!`
        # before reaching the guard, with a `BoundsError` from its own rule
        # compiler — loud already, and not ours to convert.
        #
        # `guard_wins` records whether our refusal is reliably the *first* thing
        # to see the call. It is false only for `gaussian_approximation`: whether
        # Mooncake reaches the guard or gives up compiling `_newton_loop` first
        # varies by platform, the same way `ChordalGMRF` under Enzyme does (the
        # AD reference page's note). Both outcomes are errors, which is the
        # property this file exists to defend; only one of them is ours to word.
        cases = [
            (
                name = "logdetcov", guard_wins = true,
                f = θ -> logdetcov(mooncake_cholmod_gmrf(θ)),
            ),
            (
                name = "logpdf", guard_wins = true,
                f = θ -> logpdf(mooncake_cholmod_gmrf(θ), z),
            ),
            (
                name = "logpdf (default solver)", guard_wins = true,
                f = θ -> logpdf(mooncake_default_gmrf(θ), z),
            ),
            (
                name = "gaussian_approximation", guard_wins = false,
                f = θ -> logpdf(gaussian_approximation(mooncake_cholmod_gmrf(θ), obs_lik), z),
            ),
            (
                name = "var", guard_wins = true,
                f = θ -> sum(var(mooncake_cholmod_gmrf(θ))),
            ),
            (
                name = "selinv", guard_wins = true,
                f = θ -> sum(selinv(mooncake_cholmod_gmrf(θ).linsolve_cache)),
            ),
            (
                name = "backward_solve", guard_wins = true,
                f = θ -> sum(backward_solve(mooncake_cholmod_gmrf(θ).linsolve_cache, z)),
            ),
        ]

        @testset "$(case.name)" for case in cases
            # Caught rather than `@test_throws`, so that one (expensive) Mooncake
            # compilation answers both "did it raise?" and "what did it say?".
            err = try
                DifferentiationInterface.gradient(case.f, MOONCAKE, copy(θ))
                nothing
            catch e
                e
            end

            # True on every version and architecture, and the whole point of the
            # guard: an exception rather than a number, and — since the process
            # is still alive to run this line — rather than signal 11.
            @test err isa Exception

            if case.guard_wins
                # Specifically *our* refusal, not merely some `ArgumentError`:
                # see `mooncake_refusal` for why the type cannot tell them apart.
                @test mooncake_refusal(err)

                # And the message has to stay actionable. It already names
                # `ChordalGMRF` if the assertion above passed, so this pins the
                # other half of the way out. `err === nothing` must fail this as
                # a test rather than error out of it: a ReTest *error* aborts the
                # entire run, not just this file.
                msg = err === nothing ? "" : sprint(showerror, err)
                @test occursin("ForwardDiff", msg)
            end
        end

        # Refusing is specific to differentiating. The primal calls still work.
        @test logdetcov(mooncake_cholmod_gmrf(θ)) isa Real
        @test length(var(mooncake_cholmod_gmrf(θ))) == n
    end

    @testset "GMRF operations that touch no factorization are not refused" begin
        # The guard is on the operations that consume a CHOLMOD factorization,
        # not on the mere presence of a `GMRF`. `sqmahal` is a quadratic form in
        # the precision matrix — no factorization, so nothing for us to refuse.
        #
        # What is asserted is therefore that *our* guard is not what stops it,
        # plus the gradient itself wherever Mooncake gets that far. Mooncake may
        # well fail here on its own account — it has no `to_cr_tangent` for a
        # `CHOLMOD.Factor`'s `Ptr`, and on 1.10 cannot even zero one — which is
        # its business, not the guard's; pinning it would make this test a
        # tripwire for unrelated Mooncake changes.
        f(θ) = sqmahal(mooncake_cholmod_gmrf(θ), z)
        outcome = try
            DifferentiationInterface.gradient(f, MOONCAKE, copy(θ))
        catch e
            e
        end

        @test !mooncake_refusal(outcome)

        if outcome isa AbstractVector
            reference = DifferentiationInterface.gradient(f, AutoFiniteDiff(), copy(θ))
            @test outcome ≈ reference rtol = 1.0e-4
            @test !iszero(outcome)
        end
    end

    @testset "ChordalGMRF stays supported" begin
        # The other half of the guard: it must refuse the CHOLMOD-backed type
        # *without* catching the pure-Julia one on the way past. Same precision
        # matrix as the refusal tests above, so the GMRF type is the only thing
        # that differs.
        chordal_cases = [
            "logdetcov" => θ -> logdetcov(mooncake_chordal_gmrf(θ)),
            "logpdf" => θ -> logpdf(mooncake_chordal_gmrf(θ), z),
        ]

        @testset "$name" for (name, f) in chordal_cases
            grad = DifferentiationInterface.gradient(f, MOONCAKE, copy(θ))
            reference = DifferentiationInterface.gradient(f, AutoFiniteDiff(), copy(θ))
            @test grad ≈ reference rtol = 1.0e-4
            @test !iszero(grad)
        end
    end
end
