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
end
