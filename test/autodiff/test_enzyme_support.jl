# Enzyme regression coverage.
#
# Two gaps let three separate silent-wrong-gradient bugs sit in the Enzyme
# extension unnoticed:
#
#  1. Every Enzyme test in this suite is gated behind `GMRF_TEST_ENZYME`, which
#     nothing in CI, the Makefile, or the docs ever sets. Enzyme had no coverage.
#  2. The tests that did exist used tridiagonal AR(1) precisions. Those have no
#     Cholesky fill-in, so they structurally cannot see the bug where a selected
#     inverse (which lives on the *factor's* pattern) is written into a shadow
#     sized for `Q`'s pattern.
#
# Everything below therefore uses a grid Laplacian, whose Cholesky has genuine
# fill-in, and the first testset asserts that the fill-in is actually there — so
# this file fails loudly rather than quietly degrading into the AR(1) blind spot
# if the fixture is ever changed.

using GaussianMarkovRandomFields
using Distributions: logpdf, logdetcov, Poisson, Normal
using SparseArrays
using SparseArrays: getcolptr
using LinearAlgebra
using LinearSolve
using Random
using Statistics: var
import Statistics
using DifferentiationInterface
using Enzyme, FiniteDiff, ForwardDiff
using ReTest: @testset, @test, @test_throws

const GMRFs = GaussianMarkovRandomFields
const ENZYME_EXT = Base.get_extension(GMRFs, :GaussianMarkovRandomFieldsEnzyme)
const ENZYME = AutoEnzyme(; function_annotation = Enzyme.Const)

struct EnzymeTestMetadata <: GMRFMetadata end

"""
Precision of a 2D grid Laplacian. Unlike a tridiagonal AR(1) precision, its
Cholesky factor has fill-in, so the selected inverse is stored on a strictly
larger pattern than `Q`.
"""
function enzyme_grid_laplacian(m)
    I_, J_ = Int[], Int[]
    for j in 1:m, i in 1:m
        idx = (j - 1) * m + i
        i < m && (push!(I_, idx); push!(J_, idx + 1))
        j < m && (push!(I_, idx); push!(J_, idx + m))
    end
    W = sparse([I_; J_], [J_; I_], 1.0, m * m, m * m)
    return spdiagm(0 => vec(sum(W, dims = 2))) - W
end

@testset "Enzyme AD support" begin
    L = enzyme_grid_laplacian(6)
    n = size(L, 1)
    Qof(θ) = sparse(exp(θ[1]) * (L + exp(θ[2]) * I))
    θ = [log(2.0), 0.3]

    Random.seed!(42)
    z = randn(n)

    cholmod_gmrf(θ) = GMRF(θ[2] * ones(n), Qof(θ), LinearSolve.CHOLMODFactorization())
    chordal_gmrf(θ) = ChordalGMRF(θ[2] * ones(n), Qof(θ))

    @testset "fixture has Cholesky fill-in" begin
        # Without this the rest of the file silently stops testing the bug class
        # it exists for: on a matrix with no fill-in, writing a selected inverse
        # into a `Q`-shaped shadow happens to be a no-op.
        Q = Qof(θ)
        Σ = GMRFs.selinv(cholmod_gmrf(θ).linsolve_cache)
        @test nnz(sparse(Σ)) > nnz(Q)
    end

    @testset "shadow accumulation preserves the sparsity pattern" begin
        # The direct unit test for the corruption: `accumulate_on_pattern!` must
        # never resize its target. Enzyme's shadow for a `SparseMatrixCSC` is
        # positionally aligned with the primal, so a `.+=` that grows the buffers
        # (as sparse `broadcast!` does) desynchronises the two and every
        # downstream gradient read is misaligned.
        Q = Qof(θ)
        Σ = GMRFs.selinv(cholmod_gmrf(θ).linsolve_cache)
        shadow = similar(Q)
        fill!(nonzeros(shadow), 0.0)

        colptr_before = copy(getcolptr(shadow))
        rowval_before = copy(rowvals(shadow))

        ENZYME_EXT.accumulate_on_pattern!((i, j) -> Σ[i, j], shadow, Val(:full))

        @test getcolptr(shadow) == colptr_before
        @test rowvals(shadow) == rowval_before
        @test length(nonzeros(shadow)) == nnz(Q)
        # Values land where they should: every stored entry picks up Σ[i, j].
        @test all(
            shadow[i, j] ≈ Σ[i, j]
                for j in 1:n for i in rowvals(Q)[nzrange(Q, j)]
        )
    end

    # Operations with a correct Enzyme rule. Each is checked against finite
    # differences, with ForwardDiff as an independent cross-check that the
    # reference itself is trustworthy.
    poisson = ExponentialFamily(Poisson)(PoissonObservations(rand(0:4, n)))
    # Sum-to-zero. Hoisted out of the closure the way a real model defines its
    # constraints — as fixed structure, not as something rebuilt per evaluation.
    A_sum, e_sum = ones(1, n), [0.0]
    constrained_gmrf(θ) = ConstrainedGMRF(cholmod_gmrf(θ), A_sum, e_sum)
    z0 = z .- Statistics.mean(z)                     # satisfies the sum-to-zero constraint

    # A diagonal precision is not just a smaller case: its shadow stores only the
    # diagonal, so an accumulator that walks every `(i, j)` raises rather than
    # ignoring the off-band writes. That gap reached CI through a docs tutorial
    # rather than through here.
    iid_gmrf(θ) = GMRF(
        θ[2] * ones(n),
        precision_matrix(IIDModel(n); τ = exp(θ[1])),
        LinearSolve.DiagonalFactorization(),
    )

    supported = [
        "logdetcov (GMRF/CHOLMOD)" => θ -> logdetcov(cholmod_gmrf(θ)),
        "logpdf (GMRF/CHOLMOD)" => θ -> logpdf(cholmod_gmrf(θ), z),
        # `gaussian_approximation` is the operation the whole hyperparameter
        # pipeline is built on, so it is covered through both the mode and the
        # posterior logpdf.
        "gaussian_approximation → logpdf" =>
            θ -> logpdf(gaussian_approximation(cholmod_gmrf(θ), poisson), z),
        "gaussian_approximation → mean" =>
            θ -> sum(abs2, Statistics.mean(gaussian_approximation(cholmod_gmrf(θ), poisson))),
        "logpdf (GMRF/Diagonal)" => θ -> logpdf(iid_gmrf(θ), z),
        "diagonal gaussian_approximation → logpdf" =>
            θ -> logpdf(gaussian_approximation(iid_gmrf(θ), poisson), z),
        # `WorkspaceGMRF` reaches the IFT solve through `workspace_solve` rather
        # than a `LinearSolve` cache, so it exercises a different branch of the
        # rule than every case above.
        "workspace gaussian_approximation → logpdf" =>
            θ -> logpdf(
            gaussian_approximation(WorkspaceGMRF(θ[2] * ones(n), Qof(θ)), poisson), z
        ),
    ]

    # `ConstrainedGMRF` mixes a bare `Float64` (`log_constraint_correction`) with
    # heap fields, which makes it a "mixed activity" type. Enzyme only accepts
    # such a type back from a custom rule on Julia 1.12; on 1.10 and 1.11 it
    # raises `MixedReturnException: ... not presently supported` before any of
    # these rules run. That is a loud failure, and it is upstream of this package.
    if VERSION >= v"1.12"
        append!(
            supported, [
                "logpdf (ConstrainedGMRF)" => θ -> logpdf(constrained_gmrf(θ), z0),
                "constrained gaussian_approximation → logpdf" =>
                    θ -> logpdf(gaussian_approximation(constrained_gmrf(θ), poisson), z0),
                "constrained gaussian_approximation → mean" =>
                    θ -> sum(
                    abs2,
                    Statistics.mean(gaussian_approximation(constrained_gmrf(θ), poisson))
                ),
            ]
        )
    end

    @testset "$name matches finite differences" for (name, f) in supported
        reference = DifferentiationInterface.gradient(f, AutoFiniteDiff(), copy(θ))
        forward = DifferentiationInterface.gradient(f, AutoForwardDiff(), copy(θ))
        enzyme = DifferentiationInterface.gradient(f, ENZYME, copy(θ))

        @test forward ≈ reference rtol = 1.0e-5     # the reference is sane
        @test enzyme ≈ reference rtol = 1.0e-5
        # A zero gradient is the specific failure mode being guarded against:
        # `logdetcov` used to return exactly [0.0, 0.0] on every Julia version,
        # and every constrained path did the same before `MixedDuplicated`
        # arguments were handled.
        @test !all(iszero, enzyme)
    end

    @testset "likelihood hyperparameters flow through gaussian_approximation" begin
        # Marginal-likelihood optimisation differentiates the observation model's
        # own hyperparameters, not just the prior's, and those reach the result
        # only through the nested VJPs inside the IFT rule.
        y = randn(n)
        function with_sigma(ϑ)
            prior = GMRF(ϑ[2] * ones(n), Qof(ϑ), LinearSolve.CHOLMODFactorization())
            lik = ExponentialFamily(Normal)(y; σ = exp(ϑ[3]))
            return logpdf(gaussian_approximation(prior, lik), z)
        end

        ϑ = [θ[1], θ[2], log(0.7)]
        reference = DifferentiationInterface.gradient(with_sigma, AutoFiniteDiff(), copy(ϑ))
        enzyme = DifferentiationInterface.gradient(with_sigma, ENZYME, copy(ϑ))

        @test enzyme ≈ reference rtol = 1.0e-5
        # The σ component specifically — it is the one that only exists because
        # of the nested `loggrad`/`loghessian` differentiation.
        @test enzyme[3] ≈ reference[3] rtol = 1.0e-5
        @test !iszero(enzyme[3])
    end

    @testset "unsupported operations raise instead of returning a wrong number" begin
        # `var`'s tangent reaches entries of Q⁻¹ outside the selected-inverse
        # pattern. ForwardDiff can afford to redo the selected inversion in Dual
        # arithmetic and does (see `ext/forwarddiff/var.jl`); reverse mode cannot,
        # so Enzyme must refuse rather than differentiate the factorization.
        #
        # Only "it raises" is asserted, not the exception type: whether our rule
        # gets to refuse first or Enzyme's own compilation gives up earlier
        # varies by Julia version. Both are loud failures, which is the property
        # under test.
        @test_throws Exception DifferentiationInterface.gradient(
            θ -> sum(var(cholmod_gmrf(θ))), ENZYME, copy(θ)
        )

        # The refusal is specific to differentiating: a plain primal call still
        # has to work.
        @test length(var(cholmod_gmrf(θ))) == n
    end

    @testset "ChordalGMRF is either correct or raises, never silently wrong" begin
        # `ChordalGMRF` under Enzyme is not dependable: it works on some
        # combinations of Julia version, architecture and package resolution and
        # fails on others (`EnzymeNoDerivativeError` on 1.11,
        # `EnzymeNoTypeError` on x86-64 Julia 1.12 where aarch64 succeeds). So
        # assert the invariant this whole file exists for — a wrong number is
        # never returned — rather than pinning behaviour that moves underneath us.
        for (name, f) in [
                "logdetcov" => θ -> logdetcov(chordal_gmrf(θ)),
                "logpdf" => θ -> logpdf(chordal_gmrf(θ), z),
            ]
            reference = DifferentiationInterface.gradient(f, AutoFiniteDiff(), copy(θ))
            enzyme = try
                DifferentiationInterface.gradient(f, ENZYME, copy(θ))
            catch
                nothing                          # refusing is an acceptable outcome
            end
            @test enzyme === nothing || isapprox(enzyme, reference, rtol = 1.0e-5)
        end
    end

    @testset "rules reach GMRF internals through the interface, not field names" begin
        # `ChordalGMRF` stores `μ, Q, F`; the rules used to be registered on
        # `AbstractGMRF` while reading `.linsolve_cache` / `.mean`, so they threw
        # `FieldError` from inside the reverse pass. Whatever the support status
        # of a given operation, the failure must never be a missing field.
        for f in (
                θ -> logdetcov(chordal_gmrf(θ)),
                θ -> logpdf(chordal_gmrf(θ), z),
                θ -> sum(var(chordal_gmrf(θ))),
            )
            try
                DifferentiationInterface.gradient(f, ENZYME, copy(θ))
            catch err
                @test !(err isa ErrorException && occursin("has no field", err.msg))
                @test !occursin("has no field", sprint(showerror, err))
            end
        end
    end

    # The rules are thin wrappers over the helpers below, and those helpers are
    # where the subtle mistakes live — a dropped factor of two, a rewritten
    # sparsity pattern, a wrapper read through the wrong triangle. Exercising them
    # directly costs no Enzyme compilation, so the arithmetic can be pinned much
    # more tightly than by inferring it from an end-to-end gradient.
    E = ENZYME_EXT

    @testset "accumulate_on_pattern! honours each storage convention" begin
        f(i, j) = 10i + j

        # `:full` — every stored entry is its own variable, so it takes f as-is.
        full = sparse([1.0 2.0; 2.0 3.0])
        E.accumulate_on_pattern!(f, full, Val(:full))
        @test full == [12.0 14.0; 23.0 25.0]

        # `:lower` — the upper triangle is never read, and each sub-diagonal
        # entry moves two positions of the effective matrix so it doubles.
        low = sparse([1.0 1.0; 1.0 1.0])
        E.accumulate_on_pattern!(f, low, Val(:lower))
        @test low == [12.0 1.0; 1.0 + 2 * 21 23.0]

        up = sparse([1.0 1.0; 1.0 1.0])
        E.accumulate_on_pattern!(f, up, Val(:upper))
        @test up == [12.0 1.0 + 2 * 12; 1.0 23.0]

        # Structured targets store only their band and must not be written off it.
        d = Diagonal([1.0, 2.0])
        E.accumulate_on_pattern!(f, d, Val(:full))
        @test d.diag == [12.0, 24.0]

        # `ev[i]` backs both `(i, i+1)` and `(i+1, i)`, so it collects from both.
        st = SymTridiagonal([1.0, 1.0], [1.0])
        E.accumulate_on_pattern!(f, st, Val(:full))
        @test st.dv == [12.0, 23.0]
        @test st.ev == [1.0 + f(2, 1) + f(1, 2)]

        # Dense targets, for each convention — a dense `Symmetric` precision
        # unwraps to one of these.
        dense = zeros(2, 2)
        E.accumulate_on_pattern!(f, dense, Val(:full))
        @test dense == [11.0 12.0; 21.0 22.0]

        dense_low = zeros(2, 2)
        E.accumulate_on_pattern!(f, dense_low, Val(:lower))
        @test dense_low == [11.0 0.0; 2 * 21 22.0]

        dense_up = zeros(2, 2)
        E.accumulate_on_pattern!(f, dense_up, Val(:upper))
        @test dense_up == [11.0 2 * 12; 0.0 22.0]

        wrapped = Symmetric(sparse([1.0 1.0; 1.0 1.0]), :U)
        E.accumulate_on_pattern!(f, wrapped, Val(:upper))
        @test wrapped.data[1, 2] == 1.0 + 2 * 12
        @test wrapped.data[2, 1] == 1.0                # untouched lower triangle
    end

    @testset "cotangent_matrix inverts accumulate_on_pattern!" begin
        # The pair has to round-trip: writing a symmetric gradient G into a
        # triangle-stored buffer and reading it back must return G. Getting this
        # wrong is a silent factor of two, which is why both directions exist.
        G = [1.0 2.0 0.0; 2.0 3.0 4.0; 0.0 4.0 5.0]
        pattern = sparse(ones(3, 3))

        for storage in (Val(:full), Val(:lower), Val(:upper))
            buf = copy(pattern)
            fill!(nonzeros(buf), 0.0)
            E.accumulate_on_pattern!((i, j) -> G[i, j], buf, storage)
            @test Matrix(E.cotangent_matrix(buf, storage)) ≈ G
        end
    end

    @testset "zero_shadow keeps the pattern that zero() throws away" begin
        Q = Qof(θ)
        @test nnz(zero(Q)) == 0                        # the trap being avoided
        s = E.zero_shadow(Q)
        @test nnz(s) == nnz(Q)
        @test getcolptr(s) == getcolptr(Q)
        @test all(iszero, nonzeros(s))
        # Wrappers keep their uplo, so the convention survives the round trip.
        @test E.zero_shadow(Symmetric(Q, :U)).uplo == 'U'
        @test E.zero_shadow(Hermitian(Q, :L)).uplo == 'L'
        @test E.zero_shadow(Diagonal([1.0, 2.0])).diag == [0.0, 0.0]
    end

    @testset "add_shadow! never reshapes its destination" begin
        Q = Qof(θ)
        dest, src = E.zero_shadow(Q), E.zero_shadow(Q)
        fill!(nonzeros(src), 2.0)
        E.add_shadow!(dest, src)
        @test nnz(dest) == nnz(Q)                      # fast path, same pattern
        @test all(==(2.0), nonzeros(dest))

        # Mismatched patterns take the projecting path and must still not grow.
        sparser = sparse(Diagonal(fill(3.0, n)))
        dest2 = E.zero_shadow(Q)
        E.add_shadow!(dest2, sparser)
        @test nnz(dest2) == nnz(Q)
        @test dest2[1, 1] == 3.0

        # Structured destinations, and a source read through the wrong triangle
        # would be silently wrong here.
        dd = Diagonal([0.0, 0.0])
        E.add_shadow!(dd, [1.0 9.0; 9.0 2.0])
        @test dd.diag == [1.0, 2.0]

        sd = SymTridiagonal([0.0, 0.0], [0.0])
        E.add_shadow!(sd, SymTridiagonal([1.0, 2.0], [3.0]))
        @test sd.dv == [1.0, 2.0] && sd.ev == [3.0]

        v = zeros(2)
        E.add_shadow!(v, [1.0, 2.0])
        @test v == [1.0, 2.0]

        m = zeros(2, 2)
        E.add_shadow!(m, [1.0 2.0; 3.0 4.0])
        @test m == [1.0 2.0; 3.0 4.0]

        # Wrapped destinations delegate to `.data`, positionally. Built via
        # `zero_shadow` rather than `sparse(zeros(...))`, which would have no
        # stored entries at all and silently absorb the write.
        wrapped_dest = E.zero_shadow(Symmetric(sparse(ones(2, 2)), :U))
        E.add_shadow!(wrapped_dest, Symmetric(sparse([1.0 2.0; 0.0 4.0]), :U))
        @test wrapped_dest.data[1, 2] == 2.0

        st_zero = SymTridiagonal([1.0, 2.0], [3.0])
        E._zero_precision!(st_zero)
        @test all(iszero, st_zero.dv) && all(iszero, st_zero.ev)

        # A cotangent buffer wrapped in `Symmetric` is not a symmetric matrix —
        # it must be read positionally, through `.data`.
        @test E.unwrap_triangle(Symmetric(sparse([1.0 2.0; 3.0 4.0]), :U))[2, 1] == 3.0
        @test E.unwrap_triangle([1.0 2.0; 3.0 4.0])[2, 1] == 3.0
    end

    @testset "precision_storage follows the wrapper, not the GMRF type" begin
        @test E.precision_storage(cholmod_gmrf(θ)) === Val(:full)
        @test E.precision_storage(chordal_gmrf(θ)) === Val(:lower)
        @test E._storage_convention(Symmetric(Qof(θ), :U)) === Val(:upper)
        @test E._storage_convention(Hermitian(Qof(θ), :L)) === Val(:lower)
        # A `gaussian_approximation` posterior is the `Symmetric(Q, :U)` case that
        # made the first version of the rule wrong.
        @test E.precision_storage(gaussian_approximation(cholmod_gmrf(θ), poisson)) ===
            Val(:upper)
    end

    @testset "zero_gmrf_shadow zeroes each GMRF type's cotangent slots" begin
        # The shadow keeps the primal's factorization and pattern but starts at
        # zero; rebuilding one from a zeroed precision would not be positive
        # definite, which is why these copy rather than reconstruct.
        for d in (
                cholmod_gmrf(θ), chordal_gmrf(θ),
                WorkspaceGMRF(θ[2] * ones(n), Qof(θ)), constrained_gmrf(θ),
            )
            s = E.zero_gmrf_shadow(d)
            @test typeof(s) === typeof(d)
            @test all(iszero, E.shadow_mean(s))
            @test all(iszero, E._nonzeros(E.shadow_precision(s)))
        end
    end

    @testset "MetaGMRF inherits its inner GMRF's support" begin
        inner = cholmod_gmrf(θ)
        meta = MetaGMRF(inner, EnzymeTestMetadata())
        @test E.shadow_mean(meta) === E.shadow_mean(inner)
        @test E.shadow_precision(meta) === E.shadow_precision(inner)
        @test E.enzyme_selinv(meta) == E.enzyme_selinv(inner)
        @test E.enzyme_check_supported("logpdf", meta) === nothing
    end

    @testset "contract matches a dense reference for each Hessian storage" begin
        G = [1.0 2.0; 2.0 3.0]
        for H in (Diagonal([4.0, 5.0]), sparse([4.0 1.0; 1.0 5.0]), [4.0 1.0; 1.0 5.0])
            @test E.contract(G, H) ≈ sum(G .* Matrix(H))
        end
    end

    @testset "refusals name the operation and the type" begin
        # A constrained `WorkspaceGMRF` is refused by content rather than by type:
        # its `logpdf` carries a correction these rules do not implement.
        ws = WorkspaceGMRF(θ[2] * ones(n), Qof(θ), GMRFWorkspace(Qof(θ)), A_sum, e_sum)
        err = try
            E.enzyme_check_supported("logpdf", ws)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("constraints", sprint(showerror, err))
        @test occursin("WorkspaceGMRF", sprint(showerror, err))
    end
end
