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
using Distributions: logpdf, logdetcov
using SparseArrays
using SparseArrays: getcolptr
using LinearAlgebra
using LinearSolve
using Random
using Statistics: var
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
    supported = [
        "logdetcov (GMRF/CHOLMOD)" => θ -> logdetcov(cholmod_gmrf(θ)),
        "logpdf (GMRF/CHOLMOD)" => θ -> logpdf(cholmod_gmrf(θ), z),
        "logdetcov (ChordalGMRF)" => θ -> logdetcov(chordal_gmrf(θ)),
    ]

    @testset "$name matches finite differences" for (name, f) in supported
        reference = DifferentiationInterface.gradient(f, AutoFiniteDiff(), copy(θ))
        forward = DifferentiationInterface.gradient(f, AutoForwardDiff(), copy(θ))
        enzyme = DifferentiationInterface.gradient(f, ENZYME, copy(θ))

        @test forward ≈ reference rtol = 1.0e-5     # the reference is sane
        @test enzyme ≈ reference rtol = 1.0e-5
        # A zero gradient is the specific failure mode being guarded against:
        # `logdetcov` used to return exactly [0.0, 0.0] on every Julia version.
        @test !all(iszero, enzyme)
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
